import sys
import os
import logging
from unittest.mock import MagicMock

# Adjust path to include the code directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model_predict import model_predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model_predict_top3():
    # Mock pipe
    mock_pipe = MagicMock()
    mock_pipe.return_value = {"choices": [{"text": "Mock Response"}]}

    # Mock collection
    mock_collection = MagicMock()
    # Return 3 dummy results
    mock_collection.query.return_value = {
        "ids": [["id1", "id2", "id3"]],
        "documents": [["Query1", "Query2", "Query3"]],
        "metadatas": [
            [
                {"full_content": "Content 1", "full_remarks": "Remark 1"},
                {"full_content": "Content 2", "full_remarks": "Remark 2"},
                {"full_content": "Content 3", "full_remarks": "Remark 3"},
            ]
        ],
        "pd_distances": [[0.1, 0.2, 0.3]],
    }

    patient_info = "Patient Info"
    target_group = "成人"
    complaint = "Headache"

    # Call model_predict
    final_decision, top_content, top_remarks, top_query_text = model_predict(
        mock_pipe, patient_info, mock_collection, target_group, complaint
    )

    # Verification
    logger.info(f"Top Content (Type: {type(top_content)}): {top_content}")
    logger.info(f"Top Remarks (Type: {type(top_remarks)}): {top_remarks}")

    assert isinstance(top_content, list), "top_content should be a list"
    assert len(top_content) == 3, "top_content should have 3 items"
    assert top_content == ["Content 1", "Content 2", "Content 3"]

    assert isinstance(top_remarks, list), "top_remarks should be a list"
    assert len(top_remarks) == 3, "top_remarks should have 3 items"
    assert top_remarks == ["Remark 1", "Remark 2", "Remark 3"]

    logger.info(
        "Verification Successful: model_predict is using Top 3 results correctly!"
    )


if __name__ == "__main__":
    test_model_predict_top3()
