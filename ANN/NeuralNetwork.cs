using System;
using System.Collections.Generic;

namespace ANN
{
    public class NeuralNetwork
    {
        public readonly Dictionary<string, MatrixVectors> theta = new Dictionary<string, MatrixVectors>();

        private MatrixVectors Sigmoid(MatrixVectors z)
        {
            //This method does the sigmoid calculation equivalent to 1 / (1 + np.Exp(-z)) in python

            MatrixVectors activationsVector = MatrixCalculations.Exp(MatrixCalculations.BroadcastScalar(z, -1, Operation.Multiply));
            activationsVector = MatrixCalculations.BroadcastScalar(activationsVector, 1, Operation.Add);
            activationsVector = MatrixCalculations.BroadcastScalar(activationsVector, 1, Operation.DivideUnder);

            return activationsVector;
        }

        private MatrixVectors Relu(MatrixVectors z)
        {
            MatrixVectors activationsVector = MatrixCalculations.Maximum(z, 0);

            return activationsVector;
        }

        public void InitalizeParameters(int[] dims)
        {
            for(int l = 1; l < dims.Length; l++)
            {
                MatrixVectors weights = new MatrixVectors(dims[l], dims[l - 1]);
                MatrixVectors bias = new MatrixVectors(dims[l], 1);

                weights.InitializeRandom();
                bias.InitializeWithZeros();

                theta.Add("W" + l, weights);
                theta.Add("b" + l, bias);
            }
        }

        private Tuple<LinearCache, MatrixVectors> LinearForward(MatrixVectors previousLayersActivations, MatrixVectors weights, MatrixVectors bias)
        {
            ///<summary>
            /// This method runs the linear function z = MatrixMultiplication(w, A_prev) + b
            ///
            /// It returns the linear cache which holds the weights, bias and the previous layers activations
            /// along with the Z.
            ///</summary>
           
            MatrixVectors z = MatrixCalculations.MatrixElementWise(MatrixCalculations.MatrixMultiplication(weights, previousLayersActivations), bias, Operation.Add);
            LinearCache linearCache = new LinearCache(weights, bias, previousLayersActivations);

            if (z.columns != weights.rows && z.rows != previousLayersActivations.columns)
            {
                Console.WriteLine("Z is not the proper shape");
                Console.WriteLine(z.Shape());
            }

            return new Tuple<LinearCache, MatrixVectors>(linearCache, z);
        }

        private Tuple<LinearCache, MatrixVectors, MatrixVectors> ActivationsForward(MatrixVectors previousLayersActivations, MatrixVectors weights, MatrixVectors bias, Activation activation)
        {
            ///<summary>
            /// This method runs the linear function and the specified activation function
            /// to calculate the Z and A of the current layer.
            ///
            /// It returns a tuple with the cache as the first item and the final activations as
            /// the second item.
            /// 
            /// These are returned and later stored for back prop.
            ///</summary>

            Tuple<LinearCache, MatrixVectors> cache = LinearForward(previousLayersActivations, weights, bias);
            MatrixVectors z = cache.Item2;
            MatrixVectors activationsVector;
            switch (activation)
            {
                case Activation.Sigmoid:
                    activationsVector = Sigmoid(z);
                    break;
                case Activation.ReLu:
                    activationsVector = Relu(z);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
            LinearCache linearCache = cache.Item1;
            return new Tuple<LinearCache, MatrixVectors, MatrixVectors>(linearCache, z, activationsVector);
        }

        public Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> ForwardPropagation(MatrixVectors xInput, Dictionary<string, MatrixVectors> theta, int[] dims) 
        {
            ///<summary>
            /// This methods job is the calculate the activations of each layer.
            /// It uses the X layer/input layer as the first layers previous activations
            /// and uses theta/parameters to calculate the linear function for the activations.
            /// 
            /// This method utilizes the LinearForward and ActivationsForward methods
            /// to calculate the final prediction and retrieve the caches.
            /// 
            /// This method gathers the linear and z caches of every layer.
            /// It will then generate a prediction(yhat) as the final layers activations.
            /// It will return a tuple containing the linear and z caches along with yhat.
            ///</summary>
            
            List<LinearCache> linearCaches = new List<LinearCache>();
            List<MatrixVectors> z_cache = new List<MatrixVectors>();
            
            MatrixVectors previousLayersactivations = xInput;

            for(int l = 1; l < dims.Length - 1; l++)
            {
                MatrixVectors weights = theta["W" + l];
                MatrixVectors bias = theta["b" + l];
                Tuple<LinearCache, MatrixVectors, MatrixVectors> cacheAndActivation = ActivationsForward(previousLayersactivations, weights, bias, Activation.ReLu);

                LinearCache linearCache = cacheAndActivation.Item1;
                MatrixVectors z = cacheAndActivation.Item2;


                linearCaches.Add(linearCache);
                z_cache.Add(z);

                previousLayersactivations = cacheAndActivation.Item3;
            }

            MatrixVectors finalWeights = theta["W" + (dims.Length - 1).ToString()];
            MatrixVectors finalBias = theta["b" + (dims.Length - 1).ToString()];
            Tuple<LinearCache, MatrixVectors, MatrixVectors> finalLinearCacheAndActivation = ActivationsForward(previousLayersactivations, finalWeights, finalBias, Activation.Sigmoid);

            LinearCache finalLinearCache = finalLinearCacheAndActivation.Item1;
            MatrixVectors finalZ = finalLinearCacheAndActivation.Item2;

            MatrixVectors finalActivation = finalLinearCacheAndActivation.Item3;
            linearCaches.Add(finalLinearCache);
            z_cache.Add(finalZ);

            Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndActivation = new Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors>(linearCaches, z_cache, finalActivation);
            
            return cachesAndActivation;
        }
    }
}
