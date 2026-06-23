MODEL_MARKER = "<start_of_turn>model\n"
DEVELOPER_MARKER = "<start_of_turn>developer\n"
END_TURN = "<end_of_turn>"

DEFAULT_MODEL_ID = "TuwaiqAcademy/AISA-AR-FunctionCall-Think"
DEFAULT_DATASET_ID = "TuwaiqAcademy/AISA-ArabicFC"
OFFICIAL_EVAL_REPO = "TuwaiqAcademy/AISA-ArabicFC-SharedTask-Leaderboard"

DEFAULT_TASK_REPEATS = {
    "full_track_b": 2,
    "arguments": 2,
    "function_name": 1,
    "tool_decision": 1,
    "call_only": 1,
}

ID_LIKE_FIELDS = {
    "id_number",
    "iqama_number",
    "visa_number",
    "recipient_iban",
    "iban",
    "insurance_number",
    "passport_number",
    "phone",
    "phone_number",
    "national_id",
    "account_number",
    "reference_number",
    "border_number",
}

ARABIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)
