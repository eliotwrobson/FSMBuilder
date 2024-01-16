import json
from shared_utils import QuestionData

def grade(data: QuestionData) -> None:
    name = 'q'

    data['params'].pop('submission_panel_str', None)

    if name not in data['format_errors']:
        data['score'] = 1.0
        fsm_json = json.loads(data['submitted_answers'][name])
        data['params']['submission_panel_str'] = json.dumps(fsm_json, indent=4, sort_keys=True)
