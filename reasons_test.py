#!/usr/bin/env python3
"""
Fixed Analysis of LLM Reasoning Quality from Actual JSON Data
Properly parses the reasoning text from the JSON structure
"""

import json
import re
from collections import defaultdict

def load_real_json_data(filepath='output/real_data_detailed_matches.json'):
    """Load the actual detailed matching results JSON file."""
    print("🔄 Loading actual detailed matching results...")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Loaded {len(data)} records from {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return []

def extract_reasoning_from_json(data):
    """Extract reasoning text from the JSON structure."""
    reasoning_data = defaultdict(list)
    
    for record in data:
        for approach_name, mappings in record.items():
            # Each mapping has format: [target_column, confidence, reasoning_text]
            for column, mapping_info in mappings.items():
                if len(mapping_info) >= 3:
                    reasoning_text = mapping_info[2]
                    confidence = mapping_info[1]
                    
                    reasoning_data[approach_name].append({
                        'column': column,
                        'confidence': confidence,
                        'reasoning': reasoning_text
                    })
    
    return reasoning_data

def analyze_reasoning_quality(reasoning_text):
    """Analyze the quality and characteristics of reasoning text."""
    if not reasoning_text or not isinstance(reasoning_text, str):
        return {
            'length_words': 0,
            'has_risk_assessment': False,
            'has_alternatives': False,
            'uses_examples': False,
            'technical_depth': 'low'
        }
    
    # Calculate word count
    word_count = len(reasoning_text.split())
    
    # Check for risk assessment indicators
    risk_indicators = [
        'risk', 'caution', 'however', 'but', 'although', 'limitation', 
        'constraint', 'potential issue', 'not perfect', 'consideration',
        'caveat', 'warning', 'uncertain', 'ambiguous'
    ]
    has_risk = any(indicator in reasoning_text.lower() for indicator in risk_indicators)
    
    # Check for alternative analysis
    alternative_indicators = [
        'chosen over', 'compared to', 'alternative', 'other option', 
        'could also', 'instead of', 'rather than', 'versus', 'vs',
        'other columns', 'no other', 'alternatives'
    ]
    has_alternatives = any(indicator in reasoning_text.lower() for indicator in alternative_indicators)
    
    # Check for examples usage
    example_indicators = [
        'example', 'e.g.', 'such as', 'for instance', 'like', 'including',
        '"', 'demonstrates', 'shown by', 'exemplified'
    ]
    uses_examples = any(indicator in reasoning_text.lower() for indicator in example_indicators)
    
    # Determine technical depth based on content and length
    technical_indicators = [
        'semantic', 'type conversion', 'validation', 'constraint', 'schema',
        'data type', 'format', 'structure', 'mapping', 'transformation',
        'compatibility', 'alignment', 'consistency'
    ]
    
    technical_count = sum(1 for indicator in technical_indicators if indicator in reasoning_text.lower())
    
    if word_count > 50 and technical_count >= 3:
        technical_depth = 'high'
    elif word_count > 25 and technical_count >= 2:
        technical_depth = 'medium'
    else:
        technical_depth = 'low'
    
    return {
        'length_words': word_count,
        'has_risk_assessment': has_risk,
        'has_alternatives': has_alternatives,
        'uses_examples': uses_examples,
        'technical_depth': technical_depth
    }

def create_reasoning_analysis_tables(reasoning_data):
    """Create comprehensive reasoning analysis tables."""
    
    # Analyze each approach
    approach_stats = {}
    
    for approach_name, reasonings in reasoning_data.items():
        if not reasonings:
            continue
            
        # Analyze all reasoning texts for this approach
        analyses = []
        confidences = []
        
        for item in reasonings:
            analysis = analyze_reasoning_quality(item['reasoning'])
            analyses.append(analysis)
            confidences.append(item['confidence'])
        
        if not analyses:
            continue
            
        # Calculate statistics
        avg_length = sum(a['length_words'] for a in analyses) / len(analyses)
        risk_pct = sum(1 for a in analyses if a['has_risk_assessment']) / len(analyses) * 100
        alt_pct = sum(1 for a in analyses if a['has_alternatives']) / len(analyses) * 100
        example_pct = sum(1 for a in analyses if a['uses_examples']) / len(analyses) * 100
        high_tech_pct = sum(1 for a in analyses if a['technical_depth'] == 'high') / len(analyses) * 100
        avg_confidence = sum(confidences) / len(confidences)
        
        approach_stats[approach_name] = {
            'avg_length': avg_length,
            'risk_pct': risk_pct,
            'alternatives_pct': alt_pct,
            'examples_pct': example_pct,
            'high_tech_pct': high_tech_pct,
            'avg_confidence': avg_confidence,
            'total_reasonings': len(analyses)
        }
    
    # Create LaTeX table
    latex_content = """\\begin{table}
\\caption{LLM Reasoning Quality Analysis (Corrected)}
\\label{tab:corrected_reasoning_analysis}
\\begin{tabular}{lcccccc}
\\toprule
Approach & Avg Length & Risk Assessment & Alternatives & Examples & High Tech Depth & Avg Confidence \\\\
\\midrule
"""
    
    for approach, stats in approach_stats.items():
        latex_content += f"{approach} & {stats['avg_length']:.0f} words & {stats['risk_pct']:.0f}\\% & {stats['alternatives_pct']:.0f}\\% & {stats['examples_pct']:.0f}\\% & {stats['high_tech_pct']:.0f}\\% & {stats['avg_confidence']:.3f} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save table
    output_file = 'output/results/corrected_reasoning_analysis.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✅ Created corrected reasoning analysis: {output_file}")
    
    # Print summary for verification
    print("\n📊 Reasoning Analysis Summary:")
    for approach, stats in approach_stats.items():
        print(f"   • {approach}:")
        print(f"     - Average reasoning length: {stats['avg_length']:.0f} words")
        print(f"     - Risk assessment: {stats['risk_pct']:.0f}%")
        print(f"     - Alternative analysis: {stats['alternatives_pct']:.0f}%")
        print(f"     - Uses examples: {stats['examples_pct']:.0f}%")
        print(f"     - High technical depth: {stats['high_tech_pct']:.0f}%")
        print(f"     - Average confidence: {stats['avg_confidence']:.3f}")
        print(f"     - Total reasonings analyzed: {stats['total_reasonings']}")
    
    return approach_stats

def create_detailed_comparison_table(reasoning_data):
    """Create a detailed side-by-side comparison of reasoning examples."""
    
    # Find sample reasonings from each approach
    sample_reasonings = {}
    
    for approach_name, reasonings in reasoning_data.items():
        if reasonings:
            # Get the longest reasoning as an example
            longest = max(reasonings, key=lambda x: len(x['reasoning'].split()))
            sample_reasonings[approach_name] = {
                'column': longest['column'],
                'confidence': longest['confidence'],
                'reasoning': longest['reasoning'][:300] + '...' if len(longest['reasoning']) > 300 else longest['reasoning']
            }
    
    # Create comparison table
    latex_content = """\\begin{table}
\\caption{Sample Reasoning Comparison}
\\label{tab:sample_reasoning_comparison}
\\begin{tabular}{p{2cm}p{6cm}p{6cm}}
\\toprule
Aspect & GPT-5 mini & GPT-4o mini \\\\
\\midrule
"""
    
    # TODO: find the correct approaches by name
    if len(sample_reasonings) >= 2:
        approaches = list(sample_reasonings.keys())
        approach1, approach2 = approaches[0], approaches[1]
        
        latex_content += f"Column & {sample_reasonings[approach1]['column']} & {sample_reasonings[approach2]['column']} \\\\\n"
        latex_content += f"Confidence & {sample_reasonings[approach1]['confidence']:.3f} & {sample_reasonings[approach2]['confidence']:.3f} \\\\\n"
        latex_content += f"Reasoning & {sample_reasonings[approach1]['reasoning']} & {sample_reasonings[approach2]['reasoning']} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    output_file = 'output/results/sample_reasoning_comparison.tex'
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✅ Created sample reasoning comparison: {output_file}")

def main():
    """Main analysis function."""
    print("🔄 Starting corrected LLM reasoning analysis...\n")
    
    # Load the actual JSON data
    data = load_real_json_data()
    if not data:
        print("❌ No data loaded, exiting.")
        return
    
    # Extract reasoning data
    reasoning_data = extract_reasoning_from_json(data)
    
    if not reasoning_data:
        print("❌ No reasoning data extracted, exiting.")
        return
    
    print(f"📊 Extracted reasoning data for {len(reasoning_data)} approaches")
    
    # Create analysis tables
    approach_stats = create_reasoning_analysis_tables(reasoning_data)
    create_detailed_comparison_table(reasoning_data)
    
    print("\n✅ Corrected reasoning analysis complete!")
    print("\n📝 Generated files:")
    print("   • output/results/corrected_reasoning_analysis.tex")
    print("   • output/results/sample_reasoning_comparison.tex")

if __name__ == "__main__":
    main()