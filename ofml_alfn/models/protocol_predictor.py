#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ofml_alfn.utils.protocol_types import (
    ExecutionTrace,
    ProcessExecutionRecord,
    ProcessSpec,
    ProtocolSpec,
)


def _as_module_dict(modules: Mapping[str, nn.Module] | nn.ModuleDict) -> nn.ModuleDict:
    if isinstance(modules, nn.ModuleDict):
        return modules

    out = nn.ModuleDict()
    for key, module in modules.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Module registry keys must be non-empty strings, got {key!r}")
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Module registry values must be nn.Module instances, got {type(module)} for key {key!r}"
            )
        out[key] = module
    if len(out) == 0:
        raise ValueError("Module registry must be non-empty")
    return out


def _infer_device_from_module(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        try:
            return next(module.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _to_2d_tensor(
    x: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=dtype, device=device)
    else:
        x = x.to(dtype=dtype, device=device)

    if x.ndim == 0:
        x = x.view(1, 1)
    elif x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor after normalization, got shape {tuple(x.shape)}")
    return x


def _extract_process_output_dict(
    output_obj: Any,
    *,
    process: ProcessSpec,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Normalize various process-module outputs into a dict keyed by process.output_keys.

    Supported return forms
    ----------------------
    1. dict[str, Tensor-like]
    2. Tensor-like
       - if len(output_keys) == 1: mapped to that single key
       - if len(output_keys) > 1 and last dim matches len(output_keys):
         split column-wise
    """
    if isinstance(output_obj, Mapping):
        out: Dict[str, torch.Tensor] = {}
        missing = [k for k in process.output_keys if k not in output_obj]
        if missing:
            raise KeyError(
                f"Process {process.process_id!r} returned a dict missing required output keys: {missing}"
            )

        for key in process.output_keys:
            tensor = _to_2d_tensor(output_obj[key], dtype=dtype, device=device)
            if tensor.shape[0] != batch_size:
                if tensor.shape[0] == 1 and batch_size > 1:
                    tensor = tensor.expand(batch_size, *tensor.shape[1:])
                else:
                    raise ValueError(
                        f"Output for key {key!r} in process {process.process_id!r} has "
                        f"batch size {tensor.shape[0]}, expected {batch_size}"
                    )
            out[key] = tensor
        return out

    tensor = _to_2d_tensor(output_obj, dtype=dtype, device=device)
    if tensor.shape[0] != batch_size:
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size, *tensor.shape[1:])
        else:
            raise ValueError(
                f"Process {process.process_id!r} returned batch size {tensor.shape[0]}, "
                f"expected {batch_size}"
            )

    if len(process.output_keys) == 1:
        return {process.output_keys[0]: tensor}

    if tensor.shape[1] != len(process.output_keys):
        raise ValueError(
            f"Process {process.process_id!r} has {len(process.output_keys)} output_keys "
            f"but returned tensor shape {tuple(tensor.shape)}"
        )

    return {
        key: tensor[:, i : i + 1]
        for i, key in enumerate(process.output_keys)
    }


@dataclass(frozen=True)
class ProtocolForwardResult:
    """
    Result of executing one protocol on a batch of conditions.
    """

    protocol_id: str
    target: torch.Tensor
    trace: ExecutionTrace
    process_outputs: Dict[str, Dict[str, torch.Tensor]]
    final_outputs: Dict[str, torch.Tensor]


class ProtocolPredictor(nn.Module):
    """
    Minimal protocol-aware predictor.

    Core behavior
    -------------
    - Receives a protocol graph as ProtocolSpec.
    - Resolves process modules by process.module_key.
    - Executes the protocol in topological order.
    - Returns the target output plus an execution trace.

    Module interface
    ----------------
    Each process module may support any of the following calling conventions:

    Preferred:
        module.forward_process(
            process=process,
            inputs=input_dict,
            condition_x=condition_x,
            process_outputs=all_previous_outputs,
        )

    Accepted fallbacks:
        module.forward(
            process=process,
            inputs=input_dict,
            condition_x=condition_x,
            process_outputs=all_previous_outputs,
        )
        module(process=..., inputs=..., condition_x=..., process_outputs=...)
        module(inputs)
        module(input_tensor)

    Output interface
    ----------------
    A process module may return:
    1. dict mapping each output_key to a tensor
    2. tensor
       - mapped to the sole output key if there is only one
       - split across columns if there are multiple output keys
    """

    def __init__(
        self,
        modules: Mapping[str, nn.Module] | nn.ModuleDict,
        *,
        dtype: torch.dtype = torch.float32,
        strict_registry: bool = True,
    ) -> None:
        super().__init__()
        self.modules_by_key = _as_module_dict(modules)
        self.default_dtype = dtype
        self.strict_registry = bool(strict_registry)

    def list_module_keys(self) -> Tuple[str, ...]:
        return tuple(self.modules_by_key.keys())

    def has_module_for_process(self, process: ProcessSpec) -> bool:
        return process.module_key in self.modules_by_key

    def get_module_for_process(self, process: ProcessSpec) -> nn.Module:
        key = process.module_key
        if key not in self.modules_by_key:
            available = sorted(self.modules_by_key.keys())
            raise KeyError(
                f"No module found for process {process.process_id!r} with module_key={key!r}. "
                f"Available keys: {available}"
            )
        return self.modules_by_key[key]

    def _stack_protocol_inputs(
        self,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        x = _to_2d_tensor(condition_x, dtype=dtype, device=device)
        if x.shape[1] != len(protocol.condition_keys):
            raise ValueError(
                f"Protocol {protocol.protocol_id!r} expects {len(protocol.condition_keys)} "
                f"condition features, got input shape {tuple(x.shape)}"
            )
        return {
            key: x[:, i : i + 1]
            for i, key in enumerate(protocol.condition_keys)
        }

    def _build_process_inputs(
        self,
        process: ProcessSpec,
        *,
        protocol_inputs: Mapping[str, torch.Tensor],
        process_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        inputs: Dict[str, torch.Tensor] = {}

        for key in process.input_keys:
            if key in protocol_inputs:
                inputs[key] = protocol_inputs[key]
                continue

            found = False
            for parent_id in process.parent_ids:
                parent_out = process_outputs.get(parent_id, {})
                if key in parent_out:
                    inputs[key] = parent_out[key]
                    found = True
                    break

            if not found:
                raise KeyError(
                    f"Could not resolve input key {key!r} for process {process.process_id!r}. "
                    f"Checked protocol inputs and parent outputs from {process.parent_ids}."
                )
        return inputs

    def _call_process_module(
        self,
        module: nn.Module,
        *,
        process: ProcessSpec,
        inputs: Mapping[str, torch.Tensor],
        condition_x: torch.Tensor,
        process_outputs: Mapping[str, Mapping[str, torch.Tensor]],
    ) -> Any:
        if hasattr(module, "forward_process"):
            return module.forward_process(
                process=process,
                inputs=inputs,
                condition_x=condition_x,
                process_outputs=process_outputs,
            )

        try:
            return module(
                process=process,
                inputs=inputs,
                condition_x=condition_x,
                process_outputs=process_outputs,
            )
        except TypeError:
            pass

        if len(inputs) == 1:
            sole_tensor = next(iter(inputs.values()))
            try:
                return module(sole_tensor)
            except TypeError:
                pass

        return module(inputs)

    def forward_protocol(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
        return_trace: bool = True,
    ) -> ProtocolForwardResult | torch.Tensor:
        if not isinstance(protocol, ProtocolSpec):
            raise TypeError(f"protocol must be ProtocolSpec, got {type(protocol)}")

        device = _infer_device_from_module(self)
        dtype = self.default_dtype

        x = _to_2d_tensor(condition_x, dtype=dtype, device=device)
        protocol_inputs = self._stack_protocol_inputs(protocol, x, dtype=dtype, device=device)

        process_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        records: list[ProcessExecutionRecord] = []

        for process_id in protocol.topological_order():
            process = protocol.get_process(process_id)

            if process.module_key not in self.modules_by_key:
                if self.strict_registry:
                    raise KeyError(
                        f"Missing module for process {process.process_id!r} "
                        f"(module_key={process.module_key!r})"
                    )
                continue

            module = self.get_module_for_process(process)
            inputs = self._build_process_inputs(
                process,
                protocol_inputs=protocol_inputs,
                process_outputs=process_outputs,
            )

            output_obj = self._call_process_module(
                module,
                process=process,
                inputs=inputs,
                condition_x=x,
                process_outputs=process_outputs,
            )
            output_dict = _extract_process_output_dict(
                output_obj,
                process=process,
                batch_size=x.shape[0],
                dtype=dtype,
                device=device,
            )

            process_outputs[process.process_id] = output_dict

            if return_trace:
                record = ProcessExecutionRecord(
                    process_id=process.process_id,
                    process_type=process.process_type,
                    inputs={k: v.detach().cpu() for k, v in inputs.items()},
                    outputs={k: v.detach().cpu() for k, v in output_dict.items()},
                    success=True,
                    message="ok",
                    metadata={
                        "module_key": process.module_key,
                        "role": process.role,
                    },
                )
                records.append(record)

        target_proc_id = protocol.target_process_id
        target_key = protocol.target_output_key

        if target_proc_id not in process_outputs:
            raise KeyError(
                f"Target process {target_proc_id!r} was not executed for protocol {protocol.protocol_id!r}"
            )
        if target_key not in process_outputs[target_proc_id]:
            raise KeyError(
                f"Target output key {target_key!r} missing from process {target_proc_id!r}"
            )

        target = process_outputs[target_proc_id][target_key]
        final_outputs = {
            f"{pid}.{k}": v
            for pid, out_dict in process_outputs.items()
            for k, v in out_dict.items()
        }

        trace = ExecutionTrace(
            protocol_id=protocol.protocol_id,
            condition_id=None,
            records=records,
            final_outputs={k: v.detach().cpu() for k, v in final_outputs.items()},
            target_value=target.detach().cpu(),
            metadata={
                "batch_size": int(x.shape[0]),
                "target_process_id": target_proc_id,
                "target_output_key": target_key,
            },
        )

        result = ProtocolForwardResult(
            protocol_id=protocol.protocol_id,
            target=target,
            trace=trace,
            process_outputs=process_outputs,
            final_outputs=final_outputs,
        )
        return result

    def forward_target(
        self,
        *,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
    ) -> torch.Tensor:
        result = self.forward_protocol(
            protocol=protocol,
            condition_x=condition_x,
            return_trace=False,
        )
        if isinstance(result, ProtocolForwardResult):
            return result.target
        if not torch.is_tensor(result):
            raise TypeError(f"Unexpected return type from forward_protocol: {type(result)}")
        return result

    def forward(
        self,
        protocol: ProtocolSpec,
        condition_x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_target(protocol=protocol, condition_x=condition_x)


def build_protocol_predictor(
    modules: Mapping[str, nn.Module] | nn.ModuleDict,
    *,
    dtype: torch.dtype = torch.float32,
    strict_registry: bool = True,
    device: Optional[torch.device | str] = None,
) -> ProtocolPredictor:
    predictor = ProtocolPredictor(
        modules=modules,
        dtype=dtype,
        strict_registry=strict_registry,
    )
    if device is not None:
        predictor = predictor.to(device=device)
    return predictor


__all__ = [
    "ProtocolForwardResult",
    "ProtocolPredictor",
    "build_protocol_predictor",
]