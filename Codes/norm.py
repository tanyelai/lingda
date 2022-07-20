import normalization.normalization_pb2 as z_normalize
import normalization.normalization_pb2_grpc as z_normalize_g
import grpc


class Normalizer:
    def __init__(self, host: str = 'localhost', port: str = '6789'):
        channel = grpc.insecure_channel(f'{host}:{port}')

        self.norm_stub = z_normalize_g.NormalizationServiceStub(channel)

    def normalize(self, text):
        response = self.norm_stub.Normalize(
            z_normalize.NormalizationRequest(input=text))
        return response.normalized_input
