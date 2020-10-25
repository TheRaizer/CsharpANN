namespace ANN
{
    public class LinearCache
    {
        public MatrixVectors weights;
        public MatrixVectors bias;
        public MatrixVectors previousLayersActivations;

        public LinearCache(MatrixVectors _weights, MatrixVectors _bias, MatrixVectors _previousLayersActivation)
        {
            weights = _weights;
            bias = _bias;
            previousLayersActivations = _previousLayersActivation;
        }
    }
}
