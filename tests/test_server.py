from server import app


def test_server():
  client = app.test_client()

  response = client.get("/?text=good")
  assert response.status_code == 200
  assert response.data == b'{"percentage":99,"sentiment":"Positive"}\n'

  response = client.get("/?text=bad")
  assert response.status_code == 200
  assert response.data == b'{"percentage":99,"sentiment":"Negative"}\n'