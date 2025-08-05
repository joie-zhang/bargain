#!/usr/bin/env python3
"""
Hook to remove max_tokens limitations from Claude Code requests
This ensures unlimited response generation capability
"""

import json
import sys
import os

def remove_max_tokens_hook():
    """Remove max_tokens from request to allow unlimited generation"""
    try:
        # Read the hook context from stdin
        if not sys.stdin.isatty():
            input_data = sys.stdin.read().strip()
            if input_data:
                context = json.loads(input_data)
                
                # Track if we made any changes
                changes_made = False
                
                # Remove max_tokens from top-level request
                if 'request' in context and 'max_tokens' in context['request']:
                    del context['request']['max_tokens']
                    changes_made = True
                    
                # Remove from parameters if present
                if 'request' in context and 'parameters' in context['request']:
                    if 'max_tokens' in context['request']['parameters']:
                        del context['request']['parameters']['max_tokens']
                        changes_made = True
                
                # Remove from any nested structures
                def remove_max_tokens_recursive(obj):
                    if isinstance(obj, dict):
                        if 'max_tokens' in obj:
                            del obj['max_tokens']
                            return True
                        return any(remove_max_tokens_recursive(v) for v in obj.values())
                    elif isinstance(obj, list):
                        return any(remove_max_tokens_recursive(item) for item in obj)
                    return False
                
                if remove_max_tokens_recursive(context):
                    changes_made = True
                
                if changes_made:
                    print(f"[Hook] Removed max_tokens limitations", file=sys.stderr)
                
                # Output the modified context
                print(json.dumps(context))
                return
        
        # If no input or malformed, pass through unchanged
        if not sys.stdin.isatty():
            input_data = sys.stdin.read()
            print(input_data, end='')
        
    except Exception as e:
        print(f"[Hook Error] Failed to process max_tokens removal: {e}", file=sys.stderr)
        # Pass through original input on error
        if not sys.stdin.isatty():
            input_data = sys.stdin.read()
            print(input_data, end='')

if __name__ == "__main__":
    remove_max_tokens_hook()