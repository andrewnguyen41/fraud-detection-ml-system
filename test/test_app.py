import json
import os
import sys
from src import create_app

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

app = create_app()
app.testing = True
client = app.test_client()


def test_index_route():
    response = client.get("/")
    assert response.status_code == 200


def test_detect_fraud_route_not_fraud():
    # Mock data for testing detection
    data = {
        "Time": 0,
        "V1": "-1.3598071336738",
        "V2": "-0.0727811733098497",
        "V3": "2.53634673796914",
        "V4": "1.37815522427443",
        "V5": "-0.338320769942518",
        "V6": "0.462387777762292",
        "V7": "0.239598554061257",
        "V8": "0.0986979012610507",
        "V9": "0.363786969611213",
        "V10": "0.0907941719789316",
        "V11": "-0.551599533260813",
        "V12": "-0.617800855762348",
        "V13": "-0.991389847235408",
        "V14": "-0.311169353699879",
        "V15": "1.46817697209427",
        "V16": "-0.470400525259478",
        "V17": "0.207971241929242",
        "V18": "0.0257905801985591",
        "V19": "0.403992960255733",
        "V20": "0.251412098239705",
        "V21": "-0.018306777944153",
        "V22": "0.277837575558899",
        "V23": "-0.110473910188767",
        "V24": "0.0669280749146731",
        "V25": "0.128539358273528",
        "V26": "-0.189114843888824",
        "V27": "0.133558376740387",
        "V28": "-0.0210530534538215",
        "Amount": 149.62,
    }

    response = client.post(
        "/detect-fraud", json=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200

    response_json = response.get_json()
    assert response_json["is_fraud"] == False


def test_detect_fraud_route_fraud():
    # Mock data for testing detection
    data = {
        "Time": 7543,
        "V1": "0.329594333318222",
        "V2": "3.71288929524103",
        "V3": "-5.77593510831666",
        "V4": "6.07826550560828",
        "V5": "1.66735901311948",
        "V6": "-2.42016841351562",
        "V7": "-0.812891249491333",
        "V8": "0.133080117970748",
        "V9": "-2.21431131204961",
        "V10": "-5.13445447110633",
        "V11": "4.56072010550223",
        "V12": "-8.87374836164535",
        "V13": "-0.797483599628474",
        "V14": "-9.17716637009146",
        "V15": "-0.25702477514424",
        "V16": "-0.871688490451564",
        "V17": "1.31301362907797",
        "V18": "0.773913872552923",
        "V19": "-2.37059945059811",
        "V20": "0.269772775978284",
        "V21": "0.156617169389793",
        "V22": "-0.652450440932299",
        "V23": "-0.551572219392364",
        "V24": "-0.716521635357197",
        "V25": "1.41571661508922",
        "V26": "0.555264739787582",
        "V27": "0.530507388890912",
        "V28": "0.404474054528712",
        "Amount": 1,
    }

    response = client.post(
        "/detect-fraud", json=json.dumps(data), content_type="application/json"
    )
    assert response.status_code == 200

    response_json = response.get_json()
    assert response_json["is_fraud"] == True


# TODO add more test cases

if __name__ == "__main__":
    # Run tests if this script is executed directly
    import pytest

    pytest.main()
