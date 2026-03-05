import json
from pathlib import Path
from datetime import datetime
import logging

try:
    from .resolver import HuggingFaceDatasetResolver
    from .validator import ParquetValidator
except ImportError:
    from resolver import HuggingFaceDatasetResolver
    from validator import ParquetValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "api_config.json"
OUTPUT_DIR = SCRIPT_DIR.parent / "bronze_metadata"
BRONZE_CONTRACT_PATH = OUTPUT_DIR / "bronze_ingestion_contract.json"


def load_config():
    """Load dataset configuration from api_config.json"""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)["datasets"]


def save_bronze_contract(contract: dict):
    """
    Save the bronze ingestion contract to local filesystem.
    This serves as the metadata layer for Databricks SQL ingestion.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    with open(BRONZE_CONTRACT_PATH, "w") as f:
        json.dump(contract, f, indent=2)
    
    logger.info(f"Bronze contract saved to: {BRONZE_CONTRACT_PATH}")


def run_bronze_ingestion():
    """
    Main ingestion workflow:
    1. Query HuggingFace dataset metadata
    2. Extract exact parquet file URLs
    3. Validate parquet files
    4. Store metadata as bronze contract for Databricks SQL
    """
    resolver = HuggingFaceDatasetResolver()
    validator = ParquetValidator()

    datasets = load_config()
    
    bronze_contract = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "datasets": []
    }

    for dataset_config in datasets:
        dataset_name = dataset_config["name"]
        api_url = dataset_config["url"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")

        try:
            # Step 1: Resolve parquet file URLs from HuggingFace API
            parquet_files = resolver.resolve(api_url)
            
            # Step 2: Validate each parquet file
            validated_files = []
            for file_info in parquet_files:
                metadata = validator.validate_remote_parquet(
                    url=file_info["url"],
                    filename=file_info["filename"]
                )
                
                # Merge file info with validation metadata
                validated_files.append({**file_info, **metadata})
            
            # Step 3: Build dataset entry for bronze contract
            dataset_entry = {
                "dataset_name": dataset_name,
                "api_endpoint": api_url,
                "file_count": len(validated_files),
                "total_rows": sum(f.get("num_rows", 0) for f in validated_files),
                "files": validated_files
            }
            
            bronze_contract["datasets"].append(dataset_entry)
            
            logger.info(
                f"✓ Dataset {dataset_name}: "
                f"{len(validated_files)} file(s), "
                f"{dataset_entry['total_rows']:,} total rows"
            )

        except Exception as e:
            logger.error(f"✗ Failed to process {dataset_name}: {str(e)}")
            bronze_contract["datasets"].append({
                "dataset_name": dataset_name,
                "api_endpoint": api_url,
                "status": "failed",
                "error": str(e)
            })

    # Step 4: Save bronze contract to local filesystem
    save_bronze_contract(bronze_contract)
    
    logger.info("\n" + "="*60)
    logger.info("Bronze ingestion complete!")
    logger.info(f"Total datasets processed: {len(bronze_contract['datasets'])}")
    logger.info(f"Contract location: {BRONZE_CONTRACT_PATH}")
    logger.info("="*60)


if __name__ == "__main__":
    run_bronze_ingestion()