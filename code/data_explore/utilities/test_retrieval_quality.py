import logging
import sys
import os

# Ensure we can import from the current directory
sys.path.append(os.getcwd())

from .model_init import model_init

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def test_retrieval():
    logger.info("Initializing model and vector DB...")

    try:
        pipe, collection = model_init()
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return

    # Categorized test queries
    test_cases = {
        "頭頸部外傷": [
            "頭部鈍傷",
            "頭很痛，撞到了",
            "眼睛被球打到，看不清楚",
            "鼻子被打歪了，流鼻血",
            "脖子被刀割傷",
        ],
        "胸腹背部外傷": [
            "車禍胸口撞到方向盤，很喘",
            "肚子被刀子捅了",
            "背部從高處跌落撞擊",
        ],
        "肢體外傷": [
            "手指被切斷了",
            "腳扭到了，腫起來",
            "大腿骨折，變形",
            "手腕割腕自殺",
        ],
        "其他外傷": ["全身燒傷，很痛", "被性侵", "被家暴，全身都是傷"],
        "非外傷 (內科)": [
            "胸悶，冒冷汗",
            "肚子痛，狂拉肚子",
            "頭暈，想吐",
            "發高燒，全身無力",
        ],
    }

    logger.info("Starting comprehensive retrieval test (n_results=5)...")

    for category, queries in test_cases.items():
        logger.info(f"\n=== {category} ===")
        for query in queries:
            logger.info(f"\nQuery: {query}")
            results = collection.query(
                query_texts=[query],
                n_results=5,  # Increased to 5 to see if the correct answer is in top 5
            )

            if results["documents"]:
                for i in range(len(results["documents"][0])):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    cat = meta.get("category", "Unknown")
                    sub = meta.get("sub_category", "Unknown")
                    logger.info(f"  [{i + 1}] {doc} ({sub})")
            else:
                logger.warning("  No results found.")


if __name__ == "__main__":
    test_retrieval()
