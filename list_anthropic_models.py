#!/usr/bin/env python3
"""
List available Anthropic models via API.
Requires ANTHROPIC_API_KEY environment variable to be set.
"""

import anthropic
import os
import sys

def list_anthropic_models():
    """List all available Anthropic models."""
    
    # Check for API key
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("\nTo set it, run:")
        print('export ANTHROPIC_API_KEY="your-api-key-here"')
        sys.exit(1)
    
    # Initialize client
    client = anthropic.Anthropic()
    
    # List models
    models = client.models.list(limit=20)
    
    print('Available Anthropic Models:')
    print('=' * 60)
    
    # Display models
    model_list = []
    if hasattr(models, 'data'):
        model_list = models.data
    else:
        model_list = models
        
    for i, model in enumerate(model_list, 1):
        model_id = model.id if hasattr(model, 'id') else str(model)
        print(f'{i:2}. {model_id}')
        
        # Parse model name for better display
        if 'opus' in model_id:
            model_type = 'Opus (Most capable)'
        elif 'sonnet' in model_id:
            model_type = 'Sonnet (Balanced)'
        elif 'haiku' in model_id:
            model_type = 'Haiku (Fast & efficient)'
        else:
            model_type = 'Unknown type'
            
        print(f'    Type: {model_type}')
        
        # Extract date if present
        if len(model_id.split('-')) > 0:
            date_part = model_id.split('-')[-1]
            if date_part.isdigit() and len(date_part) == 8:
                year = date_part[:4]
                month = date_part[4:6]
                day = date_part[6:8]
                print(f'    Released: {year}-{month}-{day}')
        print()
        
    print(f'\nTotal models available: {len(model_list)}')

if __name__ == "__main__":
    list_anthropic_models()