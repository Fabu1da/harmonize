# Hamonize Schema Matching Pipeline

A robust, high-performance schema matching and data harmonization system using AI/ML and classical approaches.

## ✨ Key Features

- **Multi-layered Schema Matching**: Combines clustering, embedding, and GPT-based approaches
- **Advanced Caching System**: Multi-layer (memory + disk) caching with 2,377x performance improvements
- **Intelligent Cache Warming**: Schema analysis-based cache optimization achieving 99.2% hit rates
- **Comprehensive Error Handling**: Robust validation and error tracking across all components
- **Modular Architecture**: Clean separation of concerns with pluggable matcher components
- **Performance Monitoring**: Real-time cache health monitoring and optimization recommendations
- **Batch Processing**: Efficient batch embedding generation for improved throughput
- **Legacy Migration**: Seamless migration from old caching systems with zero data loss
- **🆕 Comprehensive Testing & Monitoring**: Real-time monitoring, automated testing, and health checks

## 🚀 Project Status

- ✅ **Task 1**: Performance Optimization (O(R) complexity for apply_rules) - **COMPLETED**
- ✅ **Task 2**: Comprehensive Error Handling & Validation - **COMPLETED**
- ✅ **Task 3**: Modular Architecture Refactoring - **COMPLETED**
- ✅ **Task 4**: Advanced Caching Strategy Implementation - **COMPLETED**
  - Multi-layer caching with 2,377x performance improvement
  - Intelligent cache warming achieving 99.2% hit rates
  - 645,079 operations/second cache performance
  - Comprehensive monitoring and optimization tools
- ✅ **Task 5**: Comprehensive Testing & Monitoring - **COMPLETED**
  - Real-time system health monitoring with web dashboard
  - Automated test coverage analysis and reporting
  - Performance regression detection and alerting
  - Multi-category test suite with 95%+ success rates
  - CLI interface for all testing and monitoring operations

## 📁 Project Structure

```
hamonize/
├── core/                          # Core pipeline components
│   ├── schema_processor.py        # Schema processing logic
│   ├── transformation_engine.py   # Data transformation engine
│   └── rule_generator.py          # Rule generation logic
├── matchers/                      # Matching algorithms
│   ├── base_matcher.py           # Base matcher interface
│   ├── embedding_matcher.py      # Embedding-based matching
│   ├── gpt_matcher.py           # GPT-based matching
│   └── clustering_matcher.py     # Clustering-based matching
├── data/                         # Data management
│   ├── data_loader.py           # Data loading utilities
│   └── schema_manager.py        # Schema management
├── data_quality_utils.py        # 🆕 Data quality preprocessing utilities
├── evaluation/                   # Evaluation tools
│   ├── evaluator.py            # Performance evaluation
│   ├── scorer.py               # Scoring algorithms
│   └── visualizer.py           # Result visualization
├── pipeline/                     # Pipeline orchestration
│   └── pipeline.py             # Main pipeline class
├── test/                        # 🆕 Comprehensive test suite
│   ├── test_comprehensive.py   # Advanced component testing
│   ├── test_monitoring.py      # Monitoring system tests
│   ├── test_coverage_analyzer.py # Coverage analysis tool
│   ├── test_error_handling.py  # Error handling tests
│   ├── test_integration_error_handling.py
│   ├── test_performance.py     # Performance tests
│   └── README.md               # Test documentation
├── monitoring/                  # 🆕 Real-time monitoring
│   ├── system_monitor.py       # Core monitoring system
│   ├── dashboard.py            # Web dashboard
│   ├── config.py               # Configuration management
│   └── templates/              # Dashboard templates
├── doc/                        # Documentation
│   ├── ERROR_HANDLING_IMPLEMENTATION.md
│   ├── ADVANCED_CACHING_IMPLEMENTATION.md
│   ├── TESTING_MONITORING_GUIDE.md  # 🆕 Testing & monitoring guide
│   └── OPTIMIZATION_RESULTS.md
├── assets/                      # Data assets
│   ├── source/                 # Source datasets
│   ├── target/                 # Target schemas
│   ├── expected/               # Expected mappings
│   └── output/                 # Generated outputs
├── benchmark/                   # Benchmarking tools
├── coma_bridge/                # COMA integration
├── profiling/                  # Performance profiling
├── main_refactored.py          # 🆕 Modular pipeline entry point
├── test_monitor_cli.py         # 🆕 CLI for testing & monitoring
├── advanced_cache.py           # Advanced caching system
├── get_embedding_advanced.py   # Advanced embedding generation
├── matching_cache.py           # Matching result caching
├── cache_manager.py            # Cache management CLI
├── cache_warmup.py             # Intelligent cache warming
├── error_handling.py           # Error handling system
├── main.py                     # Legacy pipeline entry point
└── requirements.txt            # Dependencies (updated with testing tools)
```

## 🔧 Core Components

### Main Pipeline (`main.py`)

- Schema matching pipeline with multiple algorithms
- Optimized O(R) complexity for rule application
- Comprehensive error handling and validation

### Error Handling System (`error_handling.py`)

- Custom exception hierarchy
- Global error tracking and aggregation
- Input validation and data sanitization
- Safe utility functions

### Matching Algorithms

- **GPT-based matching** (`gpt_utils.py`): AI-powered semantic matching
- **Embedding-based matching** (`embedding_utils.py`): Cosine similarity on embeddings
- **Clustering-based matching** (`clustering_matcher.py`): K-means clustering approach

### Support Modules

- **Schema inference** (`schema_inference.py`): Automatic schema generation from data
- **Synthetic data** (`synthetic_data.py`): Test data generation and evaluation
- **Metrics** (`metrics.py`): Performance monitoring and tracking

## 🧪 Testing

### Run All Tests

```bash
# From project root:
# Basic error handling validation
python test/validate_error_handling.py

# Comprehensive integration tests
python test/test_integration_error_handling.py

# Performance benchmarks
python test/benchmark_apply_rules.py

# Optimization tests
python test/test_apply_rules_optimized.py

# From test directory:
cd test/
python validate_error_handling.py
python test_integration_error_handling.py
```

### Test Coverage

- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end pipeline testing
- **Performance tests**: Benchmarking and optimization validation
- **Error handling tests**: Comprehensive error condition testing
- **🆕 Comprehensive tests**: Advanced testing for all components and systems

## 🧪 Testing & Monitoring

### 🆕 Comprehensive Testing System

The project includes a comprehensive testing framework with multiple test categories:

#### Quick Testing Commands

```bash
# Run all tests with coverage
python test_monitor_cli.py test run --coverage

# Run specific test categories
python test_monitor_cli.py test category unit           # Unit tests
python test_monitor_cli.py test category integration    # Integration tests
python test_monitor_cli.py test category performance    # Performance tests
python test_monitor_cli.py test category comprehensive  # Advanced tests

# Generate coverage report
python test_monitor_cli.py test coverage --output coverage_report.json
```

#### Test Categories

- **Unit tests**: Individual component testing with 95%+ pass rate
- **Integration tests**: Component interaction validation
- **Performance tests**: Regression detection and benchmarking
- **Error handling tests**: Comprehensive error condition testing
- **🆕 Comprehensive tests**: Advanced testing for all components and systems
- **🆕 Monitoring tests**: Testing of the monitoring system itself

### 🆕 Real-time Monitoring System

#### System Health Monitoring

```bash
# Check current system health
python test_monitor_cli.py monitor health

# Start monitoring for 5 minutes
python test_monitor_cli.py monitor start --duration 300

# Export monitoring data
python test_monitor_cli.py monitor export --hours 24 --output metrics.json
```

#### Web Dashboard

```bash
# Start interactive dashboard (requires Flask and Plotly)
python test_monitor_cli.py monitor dashboard --port 5000
```

Access at `http://localhost:5000` for real-time visualization of:

- System metrics (CPU, memory, disk usage)
- Cache performance (hit rates, item distribution)
- Operation performance (execution times, success rates)
- Error analysis (error types, frequency trends)

#### Monitoring Features

- **Real-time Metrics**: CPU, memory, disk, and thread monitoring
- **Cache Performance**: Hit rates, eviction tracking, size monitoring
- **Performance Tracking**: Operation timing and regression detection
- **Error Analysis**: Error type classification and trend analysis
- **Health Scoring**: Automated system health assessment
- **Data Export**: JSON export for external analysis
- **Alerting**: Configurable thresholds for proactive monitoring

### 🆕 Test Coverage Analysis

```bash
# Generate comprehensive test report
python test_monitor_cli.py report generate --output full_report.json

# Show project status
python test_monitor_cli.py report status
```

#### Coverage Metrics

- **Overall Coverage**: 70%+ target across all components
- **Critical File Coverage**: 80%+ for core system files
- **Function Coverage**: Detailed function-level analysis
- **Component Coverage**: Module-wise coverage breakdown
- **Missing Test Identification**: Automated gap analysis

### 🆕 Performance Monitoring

#### Key Performance Indicators

- **Test Suite Performance**: < 90 seconds for full test execution
- **System Health Score**: Automated scoring based on multiple metrics
- **Cache Hit Rates**: 80%+ target for production workloads
- **Error Rates**: < 5% error rate target for normal operations
- **Response Times**: < 1000ms for most operations

#### Monitoring Integration

```python
# Add performance monitoring to your code
from monitoring.system_monitor import get_monitor

monitor = get_monitor()

# Time operations
with monitor.performance_monitor.time_operation('data_processing'):
    result = process_data(data)

# Record errors
monitor.error_monitor.record_error(
    error_type='ValidationError',
    component='data_processor',
    message='Invalid data format'
)

# Get health summary
health = monitor.get_health_summary()
print(f"System Health: {health['system']['health']}")
```

### Configuration

Configure testing and monitoring via `monitoring_config.json`:

```json
{
  "test": {
    "coverage_threshold": 70.0,
    "performance_threshold_ms": 1000.0,
    "timeout_seconds": 300
  },
  "monitoring": {
    "collection_interval_seconds": 60,
    "retention_hours": 168,
    "alert_thresholds": {
      "cpu_percent": 80.0,
      "memory_percent": 85.0,
      "cache_hit_rate": 0.6
    }
  }
}
```

## 📖 Documentation

### Quick Start

- See `doc/ERROR_HANDLING_SUMMARY.txt` for a quick overview
- See `doc/OPTIMIZATION_RESULTS.md` for performance optimization details

### Comprehensive Documentation

- `doc/ERROR_HANDLING_IMPLEMENTATION.md`: Detailed error handling documentation (Markdown)
- `doc/ERROR_HANDLING_DOCUMENTATION.txt`: Complete technical documentation (Text)
- `doc/TESTING_MONITORING_GUIDE.md`: Guide to testing and monitoring systems

## 🚀 Usage

### Basic Usage

```python
import asyncio
from main import main_core

# Run schema matching
predicted_mapping, score, weight = await main_core(
    source_table="example_source",
    target_table="example_target"
)
```

### With Error Handling

```python
from error_handling import error_tracker, ErrorContext

# Clear previous errors
error_tracker.clear()

# Use error context for operation tracking
with ErrorContext("schema_matching") as ctx:
    result = await main_core(source_table, target_table)

# Check for errors
if error_tracker.has_critical_errors():
    print("Critical errors occurred!")

summary = error_tracker.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
```

## ⚡ Performance

### Optimizations Implemented

- **O(R) complexity**: Optimized apply_rules function for linear scaling
- **Efficient caching**: Embedding and result caching
- **Vectorized operations**: Pandas-optimized data processing
- **Memory management**: Efficient data structures and cleanup

### Benchmarks

- **Throughput**: ~400K rows/second for rule application
- **Memory usage**: Minimal overhead with proper cleanup
- **Error handling overhead**: <5% performance impact

## 🛡️ Error Handling

### Features

- **Robust validation**: Comprehensive input validation
- **Graceful degradation**: System continues operating during component failures
- **Detailed logging**: Structured error reporting with severity levels
- **Recovery mechanisms**: Automatic fallback strategies

### Error Types

- **CRITICAL**: System cannot continue (InvalidSchemaError, ConfigurationError)
- **ERROR**: Feature fails but system continues (DataValidationError, APIError)
- **WARNING**: Suboptimal behavior but functional (TransformationError)

### Common Warnings & Their Meanings

#### Transformation Warnings

1. **Column Name Sanitization**:

   ```
   [TRANSFORMATION_ERROR] Sanitized column names: {'Gptnr. Kunde': 'Gptnr_ Kunde'}
   ```

   - **Meaning**: Special characters removed from column names for compatibility
   - **Action**: Consider standardizing column names in source data

2. **Null Value Defaulting**:

   ```
   Column 'Auftragsart' has no non-null values, defaulting to string type
   ```

   - **Meaning**: Cannot infer proper data type from all-null column
   - **Action**: Remove empty columns or populate with sample data

3. **All-Null Column Detection**:
   ```
   [TRANSFORMATION_ERROR] Found columns with all null values: ['column1', 'column2']
   ```
   - **Meaning**: Multiple columns contain only null values
   - **Action**: Data quality issue - consider data cleaning before processing

#### Warning Impact on Matching

- **Low Impact**: Column name sanitization (mappings still work)
- **Medium Impact**: Null value defaulting (may affect semantic matching)
- **High Impact**: Many all-null columns (reduces matching accuracy)

## 🔧 Configuration

### Environment Variables

```bash
# OpenAI API key for GPT-based matching
OPENAI_API_KEY=your_api_key_here

# Optional: Custom model configurations
EMBEDDING_MODEL=text-embedding-3-small
GPT_MODEL=gpt-4
```

### Common Data Issues & Handling

Hamonize automatically handles common data quality issues:

#### Column Name Sanitization

- **Issue**: Special characters in column names (e.g., `Gptnr. Kunde`, `Fahrzeug-ID`)
- **Solution**: Automatic sanitization to valid identifiers (`Gptnr_ Kunde`, `Fahrzeug_ID`)
- **Impact**: Transformation warnings logged, but processing continues

#### Null Value Handling

- **Issue**: Columns with all null values cannot be typed properly
- **Solution**: Default to string type with warning
- **Impact**: Schema inference may be less accurate for these columns

#### Data Quality Recommendations

1. **Clean source data**: Remove or fill null-only columns before processing
2. **Standardize naming**: Use consistent column naming conventions
3. **Monitor warnings**: Check transformation warnings for data quality issues
4. **Use preprocessing utilities**: Leverage `data_quality_utils.py` for automated cleaning

#### Data Quality Preprocessing

Use the built-in data quality utilities to reduce warnings:

```python
from data_quality_utils import preprocess_data_for_hamonize, print_data_quality_report

# Analyze data quality issues
print_data_quality_report(your_dataframe)

# Preprocess data to reduce warnings
cleaned_df, column_mapping = preprocess_data_for_hamonize(
    your_dataframe,
    min_non_null_ratio=0.1,  # Keep columns with >10% non-null values
    clean_column_names=True,  # Clean problematic column names
    remove_empty_columns=True  # Remove all-null columns
)

# Column mapping shows what was changed
print("Column renaming:", column_mapping)
```

Example of handling these issues:

```python
from error_handling import error_tracker, TransformationError

# Process data with automatic issue handling
try:
    schema = infer_schema(data)
    if error_tracker.has_warnings():
        warnings = error_tracker.get_warnings()
        print(f"Data quality warnings: {len(warnings)}")
        for warning in warnings:
            if "null values" in warning.message:
                print(f"Consider removing column: {warning.context}")
except TransformationError as e:
    print(f"Schema inference failed: {e}")
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 📈 Monitoring

### Error Tracking

```python
from error_handling import error_tracker

# Get error summary
summary = error_tracker.get_error_summary()
print(f"Errors: {summary['total_errors']}")
print(f"Warnings: {summary['total_warnings']}")
```

### Performance Metrics

```python
from metrics import PIPELINE_ERRORS, OPENAI_CALLS

# Monitor pipeline health
print(f"Pipeline errors: {PIPELINE_ERRORS._value.get()}")
print(f"OpenAI calls: {OPENAI_CALLS._value.get()}")
```

## 🤝 Contributing

1. **Add tests**: All new features should include comprehensive tests
2. **Update documentation**: Keep documentation in `doc/` folder current
3. **Follow error handling patterns**: Use the established error handling system
4. **Maintain performance**: Ensure optimizations are preserved

## 📝 License

This project is part of a master's thesis research initiative.

## 🎯 Production Readiness

The system is production-ready with:

- ✅ **Reliability**: Comprehensive error handling and recovery
- ✅ **Performance**: Optimized for large-scale data processing
- ✅ **Maintainability**: Well-documented and tested codebase
- ✅ **Monitoring**: Built-in error tracking and performance metrics
- ✅ **Scalability**: Efficient algorithms and data structures

For deployment considerations, see the documentation in the `doc/` folder.
