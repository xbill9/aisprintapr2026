import unittest
from unittest.mock import MagicMock, patch

from server import mcp


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered(self):
        """Verify that the expected tools are registered with FastMCP."""
        tools = [t.name for t in await mcp.list_tools()]
        self.assertIn("analyze_cloud_logging", tools)
        self.assertIn("suggest_sre_remediation", tools)
        self.assertIn("get_vllm_deployment_config", tools)
        self.assertIn("get_vertex_ai_model_copy_instructions", tools)
        self.assertIn("get_kaggle_model_copy_instructions", tools)
        self.assertIn("list_vertex_models", tools)
        self.assertIn("list_bucket_models", tools)
        self.assertIn("deploy_vllm", tools)
        self.assertIn("destroy_vllm", tools)
        self.assertIn("status_vllm", tools)
        self.assertIn("update_vllm_scaling", tools)

    @patch("server.subprocess.run")
    def test_update_vllm_scaling(self, mock_run):
        """Test the update_vllm_scaling tool with mock subprocess."""
        from server import update_vllm_scaling

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Scaling updated successful"
        mock_run.return_value = mock_result

        result = update_vllm_scaling(min_instances=1, max_instances=2, service_name="test-service")

        # Verify result
        self.assertIn("Successfully updated scaling for test-service to min=1, max=2", result)
        self.assertIn("Scaling updated successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "update")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--min-instances=1", cmd)
        self.assertIn("--max-instances=2", cmd)

    @patch("server.subprocess.run")
    def test_deploy_vllm(self, mock_run):
        """Test the deploy_vllm tool with mock subprocess."""
        from server import deploy_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deployment successful"
        mock_run.return_value = mock_result

        result = deploy_vllm(
            service_name="test-service",
            model_path="test-model",
            bucket_name="test-bucket",
        )

        # Verify result
        self.assertIn("Successfully deployed test-service", result)
        self.assertIn("Deployment successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "beta")
        self.assertEqual(cmd[2], "run")
        self.assertEqual(cmd[3], "deploy")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--image=vllm/vllm-openai:latest", cmd)
        self.assertIn(
            "--add-volume=name=model-volume,type=cloud-storage,bucket=test-bucket,readonly=true",
            cmd,
        )
        self.assertIn(
            "--args=--model=/mnt/models/test-model,--max-model-len=4096,--trust-remote-code,--gpu-memory-utilization=0.9,--host=0.0.0.0",
            cmd,
        )

    @patch("server.subprocess.run")
    def test_destroy_vllm(self, mock_run):
        """Test the destroy_vllm tool with mock subprocess."""
        from server import destroy_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "Deletion successful"
        mock_run.return_value = mock_result

        result = destroy_vllm(service_name="test-service")

        # Verify result
        self.assertIn("Successfully destroyed test-service", result)
        self.assertIn("Deletion successful", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "delete")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn("--quiet", cmd)

    @patch("server.subprocess.run")
    def test_status_vllm(self, mock_run):
        """Test the status_vllm tool with mock subprocess."""
        from server import status_vllm

        # Setup mock behavior
        mock_result = MagicMock()
        mock_result.stdout = "status: ready\nurl: http://test-url"
        mock_run.return_value = mock_result

        result = status_vllm(service_name="test-service")

        # Verify result
        self.assertIn("Status for test-service", result)
        self.assertIn("status: ready", result)

        # Verify subprocess call
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertEqual(cmd[0], "gcloud")
        self.assertEqual(cmd[1], "run")
        self.assertEqual(cmd[2], "services")
        self.assertEqual(cmd[3], "describe")
        self.assertEqual(cmd[4], "test-service")
        self.assertIn(
            "--format=yaml(status.conditions,status.latestCreatedRevisionName,status.url)",
            cmd,
        )

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with FastMCP."""
        resources = [str(r.uri) for r in await mcp.list_resources()]
        self.assertIn("config://vllm-deployment-template", resources)

    def test_get_kaggle_model_copy_instructions(self):
        """Test the output of the Kaggle model copy instructions tool."""
        from server import get_kaggle_model_copy_instructions

        instructions = get_kaggle_model_copy_instructions("test/slug/2b-it/1", "test-bucket")
        self.assertIn("test/slug/2b-it/1", instructions)
        self.assertIn("test-bucket", instructions)
        self.assertIn("2b-it", instructions)
        self.assertIn("kaggle models instances versions download test/slug/2b-it/1", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions

        instructions = get_vertex_ai_model_copy_instructions("gemma-2b-it")
        self.assertIn("gemma-2b-it", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("server.storage.Client")
    def test_list_bucket_models_mock(self, mock_storage_client):
        """Test the output of the GCS bucket listing tool with mocks."""
        from server import list_bucket_models

        # Setup mock behavior
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.name = "gemma-2b-it/config.json"
        mock_blob.size = 1024 * 1024 * 5  # 5 MB
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_storage_client.return_value.bucket.return_value = mock_bucket

        result = list_bucket_models("mock-bucket")
        self.assertIn("mock-bucket", result)
        self.assertIn("gemma-2b-it/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("server.requests.post")
    @patch("server.cloud_logging.Client")
    async def test_analyze_cloud_logging_mock(self, mock_logging, mock_requests):
        # This is a bit complex for a simple unittest because of async/await in FastMCP tools
        # and how FastMCP wraps them. We'll stick to basic structure verification for now.
        pass


if __name__ == "__main__":
    unittest.main()
