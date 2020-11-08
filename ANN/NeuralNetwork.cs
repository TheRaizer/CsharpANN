using System;
using System.Collections.Generic;

namespace ANN
{
    public class NeuralNetwork
    {
        public List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> GenerateBatches(int batchCount, List<MatrixVectors> data, List<MatrixVectors> labels)
        {
            if(data.Count != labels.Count)
            {
                throw new ArgumentOutOfRangeException();
            }
            int dataIndex = 0;
            int numberOfBatches = (int)Math.Ceiling((double)data.Count / batchCount);

            List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches = new List<Tuple<List<MatrixVectors>, List<MatrixVectors>>>();

            for (int b = 0; b < numberOfBatches; b++)
            {
                List<MatrixVectors> currentInputBatch = new List<MatrixVectors>();
                List<MatrixVectors> currentLabelBatch = new List<MatrixVectors>();
                int batchCountToReach = dataIndex + batchCount;
                batchCountToReach = batchCountToReach > data.Count ? data.Count : batchCountToReach;

                for (int i = dataIndex; i < batchCountToReach; i++)
                {
                    currentInputBatch.Add(data[i]);
                    currentLabelBatch.Add(labels[i]);
                }
                Tuple<List<MatrixVectors>, List<MatrixVectors>> currentBatch = MatrixCalculations.RandomizeListUnison(currentInputBatch, currentLabelBatch);
                batches.Add(currentBatch);
                dataIndex += batchCount;
            }
            return batches;
        }

        private MatrixVectors Sigmoid(MatrixVectors z)
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

        private MatrixVectors SigmoidPrime(MatrixVectors dA, MatrixVectors Z)
        {
            ///<summary>
            /// Calculates the derivative of the cross entropy cost functionin relation to Z
            /// assuming A of the same layer as Z was calculated using the Sigmoid function.
            ///</summary>

            MatrixVectors A_prev = Sigmoid(Z);
            MatrixVectors OneMinusA_prev = MatrixCalculations.BroadcastScalar(A_prev, 1, Operation.Subtract, true);
            MatrixVectors A_prevMultipliedByOneMinusA_prev = MatrixCalculations.MatrixElementWise(A_prev, OneMinusA_prev, Operation.Multiply);
            MatrixVectors dZ = MatrixCalculations.MatrixElementWise(dA, A_prevMultipliedByOneMinusA_prev, Operation.Multiply);

            return dZ;
        }

        private MatrixVectors ReLuPrime(MatrixVectors dA, MatrixVectors Z)
        {
            ///<summary>
            /// Calculates the derivative of the cross entropy
            /// cost function in relation to Z assuming A of the same layer as Z was 
            /// calculated using the ReLu function.
            ///</summary>

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

        public Dictionary<string, MatrixVectors> InitalizeParameters(int[] dims)
        {
            ///<summary>
            /// Initializes the parameters and returns them as a dictionary
            /// the string represents the name of the parameter as W[l] or b[l].
            ///</summary>
            
            Dictionary<string, MatrixVectors> theta = new Dictionary<string, MatrixVectors>();
            for (int l = 1; l < dims.Length; l++)
            {
                MatrixVectors weights = new MatrixVectors(dims[l], dims[l - 1]);
                weights = MatrixCalculations.BroadcastScalar(weights, (float)Math.Sqrt(1 / dims[l - 1]), Operation.Multiply);
                MatrixVectors bias = new MatrixVectors(dims[l], 1);

                weights.InitializeRandom();
                bias.InitializeWithZeros();

                theta.Add("W" + l, weights);
                theta.Add("b" + l, bias);
            }
            return theta;
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

        public float ComputeCost(MatrixVectors yhat, MatrixVectors _y, float lambda, Dictionary<string, MatrixVectors> theta, int[] dims)
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

            return crossEntropyCost;
        }

        private Tuple<MatrixVectors, MatrixVectors, MatrixVectors> LinearBackward(MatrixVectors dZ, LinearCache linearCache, float lambda)
        {
            ///<summary>
            /// This method calculates the derivatives of the parameters and the 
            /// derivative of the previous layers activations all in relation to the
            /// cross entropy cost function.
            /// 
            /// This method will return the derivatives in order to calculate
            /// gradient descent as well as the other dW's and db's.
            ///</summary>

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

        private Tuple<MatrixVectors, MatrixVectors, MatrixVectors> ActivationsBackward(MatrixVectors dA, MatrixVectors Z, LinearCache linearCache, Activation activation, float lambda)
        {
            ///<summary>
            /// This method will calculate dC with respect to Z from one of the specified activations
            /// then use this dC/dZ to calculate the other derivatives.
            /// 
            /// It will then return the derivatives provided from the 
            /// LinearBackward function.
            ///</summary>
            
            MatrixVectors dZ;
            switch (activation)
            {
                case Activation.Sigmoid:
                    dZ = SigmoidPrime(dA, Z);
                    break;
                case Activation.ReLu:
                    dZ = ReLuPrime(dA, Z);
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            return LinearBackward(dZ, linearCache, lambda);
        }
        
        public Dictionary<string, MatrixVectors> BackwardPropagation(MatrixVectors Y, MatrixVectors AL, List<LinearCache> linearCache, List<MatrixVectors> zCache, float lambda)
        {
            ///<summary>
            /// This method calculates all the derivatives for each layer in the neural network.
            /// It starts by calculating derivative of the cost function
            /// with respect to the predictions yhat.
            /// After this it will go through each layer starting from the last
            /// calculating the derivative of the cost function with respect to 
            /// W, b, and A of the previous layer.
            /// 
            /// dW and db will be used for updating the parameters.
            /// 
            /// dA of the previous layer is used to calculate the previous layers derivatives.
            /// 
            /// dW is the same as dC/dW
            /// db is equivalent is dC/db etc.
            ///</summary>
            
            Dictionary<string, MatrixVectors> gradients = new Dictionary<string, MatrixVectors>();
            List<LinearCache> linearCaches = linearCache;
            List<MatrixVectors> Zs = zCache;
            int layersCount = linearCaches.Count;

            MatrixVectors YDividedByAL = MatrixCalculations.MatrixElementWise(Y, AL, Operation.Divide);
            MatrixVectors OneMinusY = MatrixCalculations.BroadcastScalar(Y, 1, Operation.Subtract, true);
            MatrixVectors OneMinusAL = MatrixCalculations.BroadcastScalar(AL, 1, Operation.Subtract, true);
            MatrixVectors OneMinusYDividedByOneMinusAL = MatrixCalculations.MatrixElementWise(OneMinusY, OneMinusAL, Operation.Divide);
            MatrixVectors dAL_P1 = MatrixCalculations.MatrixElementWise(YDividedByAL, OneMinusYDividedByOneMinusAL, Operation.Subtract);

            MatrixVectors dAL = MatrixCalculations.BroadcastScalar(dAL_P1, -1, Operation.Multiply);
            Tuple<MatrixVectors, MatrixVectors, MatrixVectors> derivatives = ActivationsBackward(dAL, Zs[layersCount - 1], linearCaches[layersCount - 1], Activation.Sigmoid, lambda);
            MatrixVectors dWL = derivatives.Item1;
            MatrixVectors dbL = derivatives.Item2;
            MatrixVectors dAPrev = derivatives.Item3;

            gradients.Add("dW" + layersCount, dWL);
            gradients.Add("db" + layersCount, dbL);

            for (int l = layersCount - 1; l > 0; l--)
            {
                Tuple<MatrixVectors, MatrixVectors, MatrixVectors> deriv = ActivationsBackward(dAPrev, Zs[l - 1], linearCaches[l - 1], Activation.ReLu, lambda);
                MatrixVectors dW = deriv.Item1;
                MatrixVectors db = deriv.Item2;
                dAPrev = deriv.Item3;
                gradients.Add("dW" + l, dW);
                gradients.Add("db" + l, db);
            }

            return gradients;
        }

        public Dictionary<string, MatrixVectors> UpdateParameters(Dictionary<string, MatrixVectors> parameters, Dictionary<string, MatrixVectors> gradients, int[] dims, float learningRate)
        {
            ///<summary>
            /// This method uses the gradients which are the derivatives dW and db of each layer
            /// to update the parameters W and b.
            /// 
            /// The reason gradient descent uses derivatives is because they represent
            /// the direction of steepest ascent on a functions surface. We minus them from the
            /// parameters because we want the steepest descent. And the learning rate just modulates
            /// how far we will move in that direction.
            /// 
            /// This method will return the updated parameters.
            ///</summary>
            
            for(int l = 1; l < dims.Length; l++)
            {
                MatrixVectors dWxLearningRate = MatrixCalculations.BroadcastScalar(gradients["dW" + l], learningRate, Operation.Multiply);
                MatrixVectors dbxLearningRate = MatrixCalculations.BroadcastScalar(gradients["db" + l], learningRate, Operation.Multiply);
                parameters["W" + l] = MatrixCalculations.MatrixElementWise(parameters["W" + l], dWxLearningRate, Operation.Subtract);
                parameters["b" + l] = MatrixCalculations.MatrixElementWise(parameters["b" + l], dbxLearningRate, Operation.Subtract);
            }

            return parameters;
        }

        public void Predict(List<MatrixVectors> inputs, List<int> trueLabels, Dictionary<string, MatrixVectors> theta, int[] dims)
        {
            ///<summary>
            /// Predicts a given input vector and theta. It uses ForwardPropagation to
            /// predict. Theta is usually the most updated theta.
            /// 
            /// When predicting we round the values. If it is greater then 0.5
            /// round to 1 else round to 0.
            /// 
            /// return the exact prediction of the network as well as printing the
            /// rounded prediction.
            ///</summary>

            float num = 0;
            float examples = inputs.Count;

            for (int e = 0; e < inputs.Count; e++)
            {
                Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> cachesAndAL = ForwardPropagation(inputs[e], theta, dims);

                Console.Write("Prediction: ");
                int predic = cachesAndAL.Item3.MatrixVector[0, 0] > 0.5 ? 1 : 0;
                Console.Write(predic);
                Console.Write("Expected: ");
                Console.Write(trueLabels[e]);
                Console.WriteLine("");
                if(predic == trueLabels[e])
                {
                    num++;
                }
            }
            float accuracy = num / examples * 100;
            Console.WriteLine("Accuracy: " + accuracy);
        }
    }
}
