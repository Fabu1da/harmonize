import random
import tempfile
import os
import pandas as pd
from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema
from COMA.valentine.valentine.run_coma_example import coma_matching


class ComaApproach(BaseApproach):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def _schema_to_csv_path(self, schema: ObjectSchema, name: str) -> str:
        """Convert an ObjectSchema to a temporary CSV file for COMA"""
        # Create sample data based on schema properties
        sample_data = {}
        
        for prop_name, prop_info in schema.properties.items():
            # Create sample values based on type
            if hasattr(prop_info, 'type'):
                if prop_info.type == 'string':
                    sample_data[prop_name] = f"sample_{prop_name}"
                elif prop_info.type == 'number' or prop_info.type == 'integer':
                    sample_data[prop_name] = 123
                elif prop_info.type == 'boolean':
                    sample_data[prop_name] = True
                else:
                    sample_data[prop_name] = f"sample_{prop_name}"
            else:
                sample_data[prop_name] = f"sample_{prop_name}"
        
        # Create DataFrame with sample row
        df = pd.DataFrame([sample_data])
        
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, prefix=f'{name}_')
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return temp_file.name
        
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        temp_files = []
        
        print(f"#########################Preparing to run COMA matching with schemas.", source_schema)
        
        try:
            # Extract schema names - check multiple possible attributes
            source_name = getattr(source_schema, 'name', None) or \
                         getattr(source_schema, 'title', None) or \
                         kwargs.get('source_name', 'source_schema')
            
            target_name = getattr(target_schema, 'name', None) or \
                         getattr(target_schema, 'title', None) or \
                         kwargs.get('target_name', 'target_schema')
            
            print(f"🔍 COMA matching: {source_name} → {target_name}")
            
            # Convert schemas to temporary CSV files
            source_csv_path = self._schema_to_csv_path(source_schema, source_name)
            target_csv_path = self._schema_to_csv_path(target_schema, target_name)
            temp_files.extend([source_csv_path, target_csv_path])
            
            print(f"   📁 Created temp files: {source_csv_path}, {target_csv_path}")
            
            # Run COMA matching with file paths
            result = coma_matching(source_csv_path, target_csv_path, source_name, target_name)

            return result
        except Exception as e:
            # Return empty predictions if COMA fails
            target_columns = list(target_schema.properties.keys())
            return {col: (None, 0.0, f"COMA failed: {str(e)}") for col in target_columns}
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        print(f"   🗑️ Cleaned up: {temp_file}")
                except Exception as e:
                    print(f"   ⚠️ Failed to clean up {temp_file}: {e}")