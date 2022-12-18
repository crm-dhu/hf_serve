from locust import HttpUser, task, constant

class LoadTest(HttpUser):
    wait_time = constant(0)
    host = "http://localhost"

    @task
    def predict_flask(self):
        request_body = {"input": "[MASK] is a music instrument."}
        self.client.post(
            "http://flask:8081/unmask", json=request_body, name="flask"
        )

    @task
    def predict_fastapi(self):
        request_body = {"input": "[MASK] is a music instrument."}
        self.client.post(
            "http://fastapi:8081/unmask", json=request_body, name="fastapi"
        )

    @task
    def predict_torchserve(self):
        request_body = {"input": "[MASK] is a music instrument."}
        self.client.post(
            "http://torchserve:8080/predictions/my_tc", json=request_body, name="torchserve"
        )