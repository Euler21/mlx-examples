from typing import Any, List, Callable


def process_module_paths(
    module: Any,
    path: str,
    processor_fn: Callable[[Any, Any, List[str]], None]
) -> None:
    """
    Common code for traversing a module structure based on dot notation with wildcard support.
    
    Args:
        module: The parent module to process
        path: Dot-separated path to the target module (e.g., 'layers.*.mlp', 'layers.*')
        processor_fn: Function to call on each matched module with (container, key, path)
    """
    parts = path.split('.')
    
    def _traverse_recursive(current: Any, remaining_parts: List[str], current_path: List[str]) -> None:
        if not remaining_parts:
            return
            
        part = remaining_parts[0]
        rest = remaining_parts[1:]
        
        if part == '*':
            # Handle lists
            if isinstance(current, list):
                for i in range(len(current)):
                    new_path = current_path + [str(i)]
                    if rest:
                        _traverse_recursive(current[i], rest, new_path)
                    else:
                        processor_fn(current, i, new_path)
            # Handle dicts
            elif isinstance(current, dict):
                for key in current.keys():
                    new_path = current_path + [str(key)]
                    if rest:
                        _traverse_recursive(current[key], rest, new_path)
                    else:
                        processor_fn(current, key, new_path)
        else:
            # Handle normal case (non-wildcard)
            key = int(part) if part.isdigit() else part
            new_path = current_path + [part]
            if rest:
                _traverse_recursive(current[key], rest, new_path)
            else:
                processor_fn(current, key, new_path)
    
    _traverse_recursive(module, parts, [])