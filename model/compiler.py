import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import time
import json


class OpType(Enum):
    CONV2D = "Conv2d"
    LINEAR = "Linear"
    CONV_TRANSPOSE2D = "ConvTranspose2d"


@dataclass
class SubarrayAllocation:
    subarray_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int


@dataclass
class WeightSplit:
    split_id: int
    start_idx: int
    end_idx: int
    allocation: SubarrayAllocation
    weight_fragment: Optional[np.ndarray] = None


@dataclass
class LayerIR:
    name: str
    op_type: OpType
    weights: Optional[np.ndarray]
    bias: Optional[np.ndarray]
    weights_shape: Optional[Tuple[int, ...]]
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    total_params: int
    splits: List[WeightSplit] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    routing_priority: int = 0


@dataclass
class CompilationUnit:
    model_name: str
    layers: List[LayerIR]
    instructions: List[str]
    timestamp: str
    array_size: int
    num_subarrays: int
    version: int = 1


class CIMCompiler:

    def __init__(self, config):
        self.config = config
        self.array_size = config.hardware_array_size
        self.num_subarrays = config.hardware_num_chips
        self.array_capacity = self.array_size * self.array_size
        self.compilation_history: List[CompilationUnit] = []
        self._current_ir: List[LayerIR] = []
        self._instructions: List[str] = []
        self._resource_table: Dict[int, List[SubarrayAllocation]] = {}

    def compile_model(self, model, input_shape, model_name: str = 'model') -> List[str]:
        parsed_layers = self._stage1_parse(model, input_shape, model_name)
        ir_layers = self._stage2_generate_ir(parsed_layers)
        instructions = self._stage3_emit_instructions(ir_layers)
        self._current_ir = ir_layers
        self._instructions = instructions
        self.compilation_history.append(CompilationUnit(
            model_name=model_name,
            layers=ir_layers,
            instructions=instructions,
            timestamp=time.strftime("%Y%m%d_%H%M%S"),
            array_size=self.array_size,
            num_subarrays=self.num_subarrays,
        ))
        return instructions

    def compile_position_predictor(self, model, state_dim: int = 4) -> List[str]:
        input_shape = (1, state_dim)
        return self.compile_model(model, input_shape, 'position_predictor')

    def compile_aean(self, model, input_dim: int = 4) -> List[str]:
        input_shape = (1, input_dim)
        return self.compile_model(model, input_shape, 'aean')

    def _stage1_parse(self, model, input_shape, model_name: str) -> List[Dict[str, Any]]:
        parsed = []
        current_shape = list(input_shape)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                layer_info = {
                    'name': f"{model_name}_{name}",
                    'op_type': OpType(type(module).__name__),
                    'weights': module.weight.data.cpu().numpy() if hasattr(module, 'weight') and module.weight is not None else None,
                    'bias': module.bias.data.cpu().numpy() if hasattr(module, 'bias') and module.bias is not None else None,
                    'weights_shape': tuple(module.weight.shape) if hasattr(module, 'weight') and module.weight is not None else None,
                    'input_shape': tuple(current_shape),
                }

                if hasattr(module, 'weight') and module.weight is not None:
                    w_shape = module.weight.shape
                    if isinstance(module, nn.Conv2d):
                        total_params = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
                        layer_info['total_params'] = total_params
                        if len(current_shape) >= 3:
                            out_h = (current_shape[2] + 2 * module.padding[0] - module.dilation[0] * (w_shape[2] - 1) - 1) // module.stride[0] + 1
                            out_w = (current_shape[3] + 2 * module.padding[1] - module.dilation[1] * (w_shape[3] - 1) - 1) // module.stride[1] + 1
                            current_shape = [current_shape[0], w_shape[0], out_h, out_w]
                        else:
                            current_shape = [current_shape[0], w_shape[0], 1, 1]
                    elif isinstance(module, nn.ConvTranspose2d):
                        total_params = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
                        layer_info['total_params'] = total_params
                        current_shape = [current_shape[0], w_shape[1], current_shape[2] * 2, current_shape[3] * 2]
                    else:
                        total_params = w_shape[0] * w_shape[1]
                        layer_info['total_params'] = total_params
                        current_shape = [current_shape[0], w_shape[0]]
                else:
                    layer_info['total_params'] = 0

                layer_info['output_shape'] = tuple(current_shape)
                parsed.append(layer_info)

            elif isinstance(module, nn.Sequential):
                for sub_name, sub_module in module.named_children():
                    if isinstance(sub_module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                        sub_info = {
                            'name': f"{model_name}_{name}_{sub_name}",
                            'op_type': OpType(type(sub_module).__name__),
                            'weights': sub_module.weight.data.cpu().numpy() if hasattr(sub_module, 'weight') and sub_module.weight is not None else None,
                            'bias': sub_module.bias.data.cpu().numpy() if hasattr(sub_module, 'bias') and sub_module.bias is not None else None,
                            'weights_shape': tuple(sub_module.weight.shape) if hasattr(sub_module, 'weight') and sub_module.weight is not None else None,
                            'input_shape': tuple(current_shape),
                        }
                        if hasattr(sub_module, 'weight') and sub_module.weight is not None:
                            w_shape = sub_module.weight.shape
                            if isinstance(sub_module, nn.Conv2d):
                                sub_info['total_params'] = w_shape[0] * w_shape[1] * w_shape[2] * w_shape[3]
                            else:
                                sub_info['total_params'] = w_shape[0] * w_shape[1]
                        else:
                            sub_info['total_params'] = 0
                        sub_info['output_shape'] = tuple(current_shape)
                        parsed.append(sub_info)

        return parsed

    def _stage2_generate_ir(self, parsed_layers: List[Dict[str, Any]]) -> List[LayerIR]:
        self._resource_table = {i: [] for i in range(self.num_subarrays)}
        ir_layers = []

        for layer_info in parsed_layers:
            splits = self._compute_weight_splits(
                layer_info.get('total_params', 0),
                layer_info.get('weights'),
            )

            default_params = {
                'IT': 100.0,
                'WCN': 1.0,
                'IEM': 0.0,
            }

            routing_priority = self._compute_routing_priority(
                layer_info.get('op_type', OpType.LINEAR),
                layer_info.get('total_params', 0),
            )

            layer_ir = LayerIR(
                name=layer_info['name'],
                op_type=layer_info['op_type'],
                weights=layer_info.get('weights'),
                bias=layer_info.get('bias'),
                weights_shape=layer_info.get('weights_shape'),
                input_shape=layer_info.get('input_shape'),
                output_shape=layer_info.get('output_shape'),
                total_params=layer_info.get('total_params', 0),
                splits=splits,
                parameters=default_params,
                routing_priority=routing_priority,
            )

            for split in splits:
                self._resource_table[split.allocation.subarray_id].append(split.allocation)

            ir_layers.append(layer_ir)

        return ir_layers

    def _compute_weight_splits(self, total_params: int,
                                weights: Optional[np.ndarray]) -> List[WeightSplit]:
        if total_params == 0:
            return []

        num_splits = max(1, (total_params + self.array_capacity - 1) // self.array_capacity)
        splits = []

        for i in range(num_splits):
            subarray_id = i % self.num_subarrays
            row_group = i // self.num_subarrays
            row_start = row_group * self.array_size
            row_end = min(row_start + self.array_size, self.array_size)

            start_idx = i * self.array_capacity
            end_idx = min((i + 1) * self.array_capacity, total_params)

            allocation = SubarrayAllocation(
                subarray_id=subarray_id,
                row_start=row_start,
                row_end=row_end,
                col_start=0,
                col_end=self.array_size,
            )

            fragment = None
            if weights is not None:
                flat_w = weights.flatten()
                if start_idx < len(flat_w):
                    fragment = flat_w[start_idx:end_idx]

            splits.append(WeightSplit(
                split_id=i,
                start_idx=start_idx,
                end_idx=end_idx,
                allocation=allocation,
                weight_fragment=fragment,
            ))

        return splits

    def _compute_routing_priority(self, op_type: OpType, total_params: int) -> int:
        base = 0
        if op_type == OpType.CONV2D:
            base = 100
        elif op_type == OpType.LINEAR:
            base = 50
        elif op_type == OpType.CONV_TRANSPOSE2D:
            base = 75
        size_bonus = min(50, total_params // 10000)
        return base + size_bonus

    def _stage3_emit_instructions(self, ir_layers: List[LayerIR]) -> List[str]:
        instructions = []
        instructions.append(f"COMPILE array_size={self.array_size} subarrays={self.num_subarrays}")
        instructions.append(f"TIMESTAMP {time.strftime('%Y-%m-%dT%H:%M:%S')}")

        for layer_ir in ir_layers:
            instructions.append(f"\nLAYER name={layer_ir.name} type={layer_ir.op_type.value}")
            instructions.append(f"  SHAPE weight={layer_ir.weights_shape} input={layer_ir.input_shape} output={layer_ir.output_shape}")
            instructions.append(f"  PARAMS count={layer_ir.total_params} splits={len(layer_ir.splits)} priority={layer_ir.routing_priority}")

            for split in layer_ir.splits:
                alloc = split.allocation
                instructions.append(
                    f"  MAP split={split.split_id} subarray={alloc.subarray_id} "
                    f"row=[{alloc.row_start}:{alloc.row_end}] col=[{alloc.col_start}:{alloc.col_end}] "
                    f"range=[{split.start_idx}:{split.end_idx}]"
                )

            instructions.append(
                f"  CONFIG IT={layer_ir.parameters['IT']:.1f} "
                f"WCN={layer_ir.parameters['WCN']:.1f} "
                f"IEM={layer_ir.parameters['IEM']:.1f}"
            )

        instructions.append(f"\nEND_COMPILE layers={len(ir_layers)}")
        return instructions

    def update_parameter(self, layer_name: str, param_name: str, value: float) -> bool:
        for layer_ir in self._current_ir:
            if layer_ir.name == layer_name:
                layer_ir.parameters[param_name] = value
                self._instructions = self._stage3_emit_instructions(self._current_ir)
                return True
        for cu in self.compilation_history:
            for layer_ir in cu.layers:
                if layer_ir.name == layer_name:
                    layer_ir.parameters[param_name] = value
                    cu.instructions = self._stage3_emit_instructions(cu.layers)
                    cu.version += 1
                    return True
        return False

    def get_layer_config(self, layer_name: str) -> Optional[Dict[str, Any]]:
        for layer_ir in self._current_ir:
            if layer_ir.name == layer_name:
                return {
                    'name': layer_ir.name,
                    'type': layer_ir.op_type.value,
                    'weights_shape': layer_ir.weights_shape,
                    'input_shape': layer_ir.input_shape,
                    'output_shape': layer_ir.output_shape,
                    'total_params': layer_ir.total_params,
                    'num_splits': len(layer_ir.splits),
                    'parameters': dict(layer_ir.parameters),
                    'routing_priority': layer_ir.routing_priority,
                    'allocations': [
                        {
                            'subarray_id': s.allocation.subarray_id,
                            'row_start': s.allocation.row_start,
                            'row_end': s.allocation.row_end,
                            'col_start': s.allocation.col_start,
                            'col_end': s.allocation.col_end,
                        }
                        for s in layer_ir.splits
                    ],
                }
        return None

    def get_resource_utilization(self) -> Dict[int, float]:
        utilization = {}
        for subarray_id, allocations in self._resource_table.items():
            used_cells = sum(
                (a.row_end - a.row_start) * (a.col_end - a.col_start)
                for a in allocations
            )
            utilization[subarray_id] = used_cells / self.array_capacity
        return utilization

    def get_ir(self) -> List[LayerIR]:
        return self._current_ir

    def get_instructions(self) -> List[str]:
        return self._instructions

    def save_instructions(self, filepath: str) -> None:
        with open(filepath, 'w') as f:
            for instr in self._instructions:
                f.write(instr + '\n')
        print(f"Instructions saved to {filepath}")

    def export_ir_json(self, filepath: str) -> None:
        ir_export = []
        for layer_ir in self._current_ir:
            layer_dict = {
                'name': layer_ir.name,
                'op_type': layer_ir.op_type.value,
                'weights_shape': list(layer_ir.weights_shape) if layer_ir.weights_shape else None,
                'input_shape': list(layer_ir.input_shape) if layer_ir.input_shape else None,
                'output_shape': list(layer_ir.output_shape) if layer_ir.output_shape else None,
                'total_params': layer_ir.total_params,
                'parameters': layer_ir.parameters,
                'routing_priority': layer_ir.routing_priority,
                'num_splits': len(layer_ir.splits),
                'allocations': [
                    {
                        'split_id': s.split_id,
                        'subarray_id': s.allocation.subarray_id,
                        'row_range': [s.allocation.row_start, s.allocation.row_end],
                        'col_range': [s.allocation.col_start, s.allocation.col_end],
                        'param_range': [s.start_idx, s.end_idx],
                    }
                    for s in layer_ir.splits
                ],
            }
            ir_export.append(layer_dict)
        with open(filepath, 'w') as f:
            json.dump(ir_export, f, indent=2)
        print(f"IR exported to {filepath}")
