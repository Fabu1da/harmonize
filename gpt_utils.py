import openai
import logging
import os
import json
import re
import functools
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def embeddings(df, schemas, n):
    # Step 1: Create embeddings for the schemas using the OpenAI API
    embedding_response = openai.embeddings.create(
        input=schemas,
        model="text-embedding-3-small"
    )
    
    # Step 2: Extract the embeddings for the schemas
    schema_embeddings = [embedding.embedding for embedding in embedding_response.data]
    
    # Step 3: Generate embeddings for the DataFrame column headers
    column_embeddings = []
    for column in df.columns:
        embedding_response = openai.embeddings.create(
            input=[column],
            model="text-embedding-3-small"
        )
        column_embeddings.append(embedding_response['data'][0]['embedding'])
    
    # Add the column embeddings to the DataFrame as 'ada_embedding'
    df['ada_embedding'] = column_embeddings
    
    # Step 4: Calculate cosine similarity between each row's embedding and the schema embeddings
    df['similarities'] = df['ada_embedding'].apply(
        lambda x: max(cosine_similarity([x], schema_embeddings)[0])  # Use max to get highest similarity
    )
    
    # Step 5: Sort by similarity and return top n results
    res = df.sort_values('similarities', ascending=False).head(n)
    print(res)
    return res

async def gpt_column_mapping(source_schema: dict, target_schema: dict) -> dict[str, str]:
    
    #TODO:
    #Use embendings 
    
    system_message = "".join([
        "You are an expert in schema matching and data integration. Your task is to match columns from a given source schema ",
        "to the most semantically similar columns in a target schema. Use both lexical similarity (string similarity) and semantic understanding. ",
        "Provide a JSON object where each target column is mapped to its most relevant source column, along with a similarity score and a description.\n\n",
        "**Instructions:**\n",
        "1. Match **each target column** to the **most similar source column** based on meaning and name similarity.\n",
        "2. If no strong match is found (low similarity), set the source column as `None`.\n",
    ])
    user_message = "".join([
        "### Source Schema:\n",
        f"{json.dumps(source_schema)}\n\n",
        "### Target Schema:\n",
        f"{json.dumps(target_schema)}\n\n",
    ])

    try:
        response = openai.chat.completions.create(  # Correct API method
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                # role: system, name: example-user OR role: user
                # role: system, name: example-assistant OR role: assistant
                {"role": "user", "content": user_message}
            ],
            functions=[
                {
                    "name": "map_schema",
                    "parameters": {
                        "type": "object",  # ✅ Fix: Expecting an object, not an array
                        "properties": {
                            "mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_column": {"type": "string"},
                                        "source_column": {"type": "string"},
                                    },
                                    "required": ["target_column", "source_column"],
                                }
                            }
                        },
                        "required": ["mappings"]
                    }
                }
            ],
            function_call={"name": "map_schema"},
            temperature=0,
        )

        # Extract and parse content correctly
        if response.choices and len(response.choices) > 0 and response.choices[0].message.function_call:
            content = response.choices[0].message.function_call.arguments
            try:
                result = json.loads(content)  # ✅ Convert JSON string to dictionary
                mappings = result.get("mappings", [])

                return {
                    item["target_column"]: item["source_column"]
                    for item in mappings
                }
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                return {target_column: None for target_column in target_schema}
        else:
            print("GPT response does not contain valid 'choices' or content.")
            return {target_column: None for target_column in target_schema}

    except Exception as e:
        print(f"Error during GPT request: {e}")
        return {target_column: None for target_column in target_schema}
