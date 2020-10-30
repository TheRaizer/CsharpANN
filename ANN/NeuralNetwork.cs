using System;
using System.Collections.Generic;

namespace ANN
{
    public class NeuralNetwork
    {
        public readonly Dictionary<string, MatrixVectors> theta = new Dictionary<string, MatrixVectors>();

        public MatrixVectors Sigmoid(MatrixVectors z)
        {
            //This method does the sigmoid calculation equivalent to 1 / (1 + np.Exp(-z)) in python

            MatrixVectors activationsVector = MatrixCalculations.Exp(MatrixCalculations.BroadcastScalar(z, -1, Operation.Multiply));
            activationsVector = MatrixCalculations.BroadcastScalar(activationsVector, 1, Operation.Add);
            activationsVector = MatrixCalculations.BroadcastScalar(activationsVector, 1, Operation.Divide, true);
            return activationsVector;
        }

        private MatrixVectors Relu(MatrixVectors z)
        {
            MatrixVectors activationsVector = MatrixCalculations.Maximum(z, 0);

            return activationsVector;
        }

        public MatrixVectors SigmoidBackward(MatrixVectors dA, MatrixVectors Z)
        {
            MatrixVectors A_prev = Sigmoid(Z);
            MatrixVectors OneMinusA_prev = MatrixCalculations.BroadcastScalar(A_prev, 1, Operation.Subtract, true);
            MatrixVectors A_prevMultipliedByOneMinusA_prev = MatrixCalculations.MatrixElementWise(A_prev, OneMinusA_prev, Operation.Multiply);
            MatrixVectors dZ = MatrixCalculations.MatrixElementWise(A_prev, A_prevMultipliedByOneMinusA_prev, Operation.Multiply);

            return dZ;
        }

        public MatrixVectors ReLuBackward(MatrixVectors dA, MatrixVectors Z)
        {
            MatrixVectors dZ = dA;
            if (!dZ.CompareShape(Z) || !Z.CompareShape(dA))
            {
                Console.WriteLine("Error");
                return null;
            }
            for(int y = 0; y < dZ.rows; y++)
            {
                for (int x = 0; x < dZ.columns; x++)
                {
                    if (Z.MatrixVector[x, y] <= 0)
                    {
                        dZ.MatrixVector[x, y] = 0;
                    }
                }
            }

            return dZ;
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
           
            MatrixVectors z = MatrixCalculations.MatrixElementWise(MatrixCalculations.Dot(weights, previousLayersActivations), bias, Operation.Add);
            LinearCache linearCache = new LinearCache(weights, bias, previousLayersActivations);

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

        public float ComputeCost(MatrixVectors yhat, MatrixVectors _y)
        {
            ///<summary>
            /// This method uses the cross entropy cost function to caculate the losses.
            /// It takes in yhat which are the predictions of the network
            /// and _y which are the true labels.
            /// 
            /// It returns a float value which is the calculated loss as well as its derivative.
            ///</summary>
            
            if (yhat.columns > 1 || _y.columns > 1 || !yhat.CompareShape(_y))
            {
                Console.WriteLine("Invalid YShape");
                return 0f;
            }

            float crossEntropyCost = 0;
            for(int y = 0; y < _y.rows; y++)
            {
                float currentYhat = yhat.MatrixVector[0, y];
                float currentY = _y.MatrixVector[0, y];
                float currentCost = (float)-(currentY * Math.Log10(currentYhat) + (1 - currentY) * Math.Log10(1 - currentYhat));
                crossEntropyCost += currentCost;
            }

            crossEntropyCost *= -1;

            return crossEntropyCost;
        }

        public Tuple<MatrixVectors, MatrixVectors, MatrixVectors> LinearBackward(MatrixVectors dZ, LinearCache linearCache)
        {
            MatrixVectors dW = MatrixCalculations.Dot(dZ, MatrixCalculations.Transpose(linearCache.previousLayersActivations));
            MatrixVectors db = MatrixCalculations.MatrixAxisSummation(dZ, 1);
            MatrixVectors dAPrev = MatrixCalculations.Dot(MatrixCalculations.Transpose(linearCache.weights), dZ);

            if (!dW.CompareShape(linearCache.weights))
            {
                Console.WriteLine("Does not have the right shape for dW");
            }
            if (!db.CompareShape(linearCache.bias))
            {
                Console.WriteLine("Does not have the right shape for db");
            }
            if (!dAPrev.CompareShape(linearCache.previousLayersActivations))
            {
                Console.WriteLine("Does not have the right shape for dAPrev");
            }
            return new Tuple<MatrixVectors, MatrixVectors, MatrixVectors>(dW, db, dAPrev);
        }

        public Tuple<MatrixVectors, MatrixVectors, MatrixVectors> ActivationsBackward(MatrixVectors dA, MatrixVectors Z, LinearCache linearCache, Activation activation)
        {
            MatrixVectors dZ;
            switch (activation)
            {
                case Activation.Sigmoid:
                    dZ = SigmoidBackward(dA, Z);
                    break;
                case Activation.ReLu:
                    dZ = ReLuBackward(dA, Z);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            return LinearBackward(dZ, linearCache);
        }
        
        public Dictionary<string, MatrixVectors> BackwardPropagation(MatrixVectors Y, MatrixVectors AL, Tuple<List<LinearCache>, List<MatrixVectors>> caches)
        {
            Dictionary<string, MatrixVectors> gradients = new Dictionary<string, MatrixVectors>();
            List<LinearCache> linearCaches = caches.Item1;
            List<MatrixVectors> Zs = caches.Item2;
            int layersCount = linearCaches.Count;

            MatrixVectors YDividedByAL = MatrixCalculations.MatrixElementWise(Y, AL, Operation.Divide);
            MatrixVectors OneMinusY = MatrixCalculations.BroadcastScalar(Y, 1, Operation.Subtract, true);
            MatrixVectors OneMinusAL = MatrixCalculations.BroadcastScalar(AL, 1, Operation.Subtract, true);
            MatrixVectors OneMinusYDividedByOneMinusAL = MatrixCalculations.MatrixElementWise(OneMinusY, OneMinusAL, Operation.Divide);
            MatrixVectors dAL_P1 = MatrixCalculations.MatrixElementWise(YDividedByAL, OneMinusYDividedByOneMinusAL, Operation.Subtract);

            MatrixVectors dAL = MatrixCalculations.BroadcastScalar(dAL_P1, -1, Operation.Multiply);
            Tuple<MatrixVectors, MatrixVectors, MatrixVectors> derivatives = ActivationsBackward(dAL, Zs[layersCount - 1], linearCaches[layersCount - 1], Activation.Sigmoid);
            MatrixVectors dWL = derivatives.Item1;
            MatrixVectors dbL = derivatives.Item2;
            MatrixVectors dAPrevL = derivatives.Item3;

            gradients.Add("dW" + layersCount, dWL);
            gradients.Add("db" + layersCount, dbL);
            gradients.Add("dA" + (layersCount - 1).ToString(), dAPrevL);

            for(int l = layersCount - 2; l < 0; l--)
            {
                Tuple<MatrixVectors, MatrixVectors, MatrixVectors> deriv = ActivationsBackward(gradients["dA" + (l + 1).ToString()], Zs[l], linearCaches[l], Activation.ReLu);
                MatrixVectors dW = deriv.Item1;
                MatrixVectors db = deriv.Item2;
                MatrixVectors dAPrev = deriv.Item3;
                gradients.Add("dW" + l, dW);
                gradients.Add("db" + l, db);
                gradients.Add("dA" + l, dAPrev);
            }

            return gradients;
        }

        public Dictionary<string, MatrixVectors> UpdateParameters(Dictionary<string, MatrixVectors> parameters, Dictionary<string, MatrixVectors> gradients, int[] dims, int learningRate)
        {
            for(int l = 1; l < dims.Length; l++)
            {
                MatrixVectors dWxLearningRate = MatrixCalculations.BroadcastScalar(gradients["dW" + l], learningRate, Operation.Multiply);
                MatrixVectors dbxLearningRate = MatrixCalculations.BroadcastScalar(gradients["dW" + l], learningRate, Operation.Multiply);

                parameters["W" + l] = MatrixCalculations.MatrixElementWise(parameters["W" + l], dWxLearningRate, Operation.Subtract);
                parameters["b" + l] = MatrixCalculations.MatrixElementWise(parameters["b" + 1], dbxLearningRate, Operation.Subtract);
            }

            return parameters;
        }
    }
}
