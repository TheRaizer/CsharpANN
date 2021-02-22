using System;
using System.Collections.Generic;

namespace ANN
{
    public class NeuralNetwork
    {
        /// <summary>
        ///     Generates mini batches for mini batch gradient descent
        /// </summary>
        /// <param name="batchCount">The number of data to be held in each mini batch.</param>
        /// <param name="data">The data that the batches will be made from.</param>
        /// <param name="labels">The true labels that relate to the data.</param>
        /// <returns></returns>
        public List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> GenerateBatches(int batchCount, List<MatrixVectors> data, List<MatrixVectors> labels)
        {
            if(data.Count != labels.Count)
            {
                throw new ArgumentOutOfRangeException();
            }
            // the start index for a mini batch to begin adding data from
            int dataIndex = 0;
            int numberOfBatches = (int)Math.Ceiling((double)data.Count / batchCount);

            List<Tuple<List<MatrixVectors>, List<MatrixVectors>>> batches = new List<Tuple<List<MatrixVectors>, List<MatrixVectors>>>();

            for (int b = 0; b < numberOfBatches; b++)
            {
                List<MatrixVectors> currentInputBatch = new List<MatrixVectors>();
                List<MatrixVectors> currentLabelBatch = new List<MatrixVectors>();

                int batchCountToReach = dataIndex + batchCount;
                batchCountToReach = batchCountToReach > data.Count ? data.Count : batchCountToReach;

                // from the data index to the batchCount needed to reach add the data to the current batch
                for (int i = dataIndex; i < batchCountToReach; i++)
                {
                    currentInputBatch.Add(data[i]);
                    currentLabelBatch.Add(labels[i]);
                }
                // randomize the batch
                Tuple<List<MatrixVectors>, List<MatrixVectors>> currentBatch = currentInputBatch.RandomizeListUnison(currentLabelBatch);
                batches.Add(currentBatch);
                dataIndex += batchCount;
            }
            return batches;
        }

        /// <summary>
        ///     This method does the sigmoid calculation equivalent to 1 / (1 + np.Exp(-z)) in python.
        /// </summary>
        /// <param name="Z">The linear function of the weights biases and previous layers activations.</param>
        /// <returns>A vector containing the non-linear sigmoid activations of the linear function z.</returns>
        private MatrixVectors Sigmoid(MatrixVectors Z)
        {
            MatrixVectors activationsVector = Z.BroadcastScalar(-1, Operation.Multiply).Exp();
            activationsVector = activationsVector.BroadcastScalar(1, Operation.Add);
            activationsVector = activationsVector.BroadcastScalar(1, Operation.Divide, true);
            return activationsVector;
        }

        /// <summary>
        /// Executes the non-linear ReLu activation function on some linear function Z.
        /// </summary>
        /// <param name="Z">The linear function of the weights biases and previous layers activations.</param>
        /// <returns>A vector containing the non-linear sigmoid activations of the linear function z.</returns>
        private MatrixVectors Relu(MatrixVectors Z)
        {
            MatrixVectors activationsVector = MatrixCalculations.Maximum(Z, 0);

            return activationsVector;
        }

        /// <summary>
        ///     Calculates the derivative of the cross entropy cost function with respect to Z
        ///     assuming A of the same layer as Z was calculated using the Sigmoid function.
        /// </summary>
        /// <param name="dA">Derivative of the cost function with respect to the activation.</param>
        /// <param name="Z">The linear function of the weights biases and previous layers activations.</param>
        /// <returns>The derivative of the cost function with respect to Z.</returns>
        private MatrixVectors SigmoidPrime(MatrixVectors dA, MatrixVectors Z)
        {
            MatrixVectors A_prev = Sigmoid(Z);
            MatrixVectors OneMinusA_prev = A_prev.BroadcastScalar(1, Operation.Subtract, true);
            MatrixVectors A_prevMultipliedByOneMinusA_prev = A_prev.MatrixElementWise(OneMinusA_prev, Operation.Multiply);
            MatrixVectors dZ = dA.MatrixElementWise(A_prevMultipliedByOneMinusA_prev, Operation.Multiply);

            return dZ;
        }

        /// <summary>
        ///     Calculates the derivative of the cross entropy
        ///     cost function with respect to to Z assuming A of the same layer as Z was 
        ///     calculated using the ReLu function.
        /// </summary>
        /// <param name="dA">Derivative of the cost function with respect to the activation.</param>
        /// <param name="Z">The linear function of the weights biases and previous layers activations.</param>
        /// <returns>The derivative of the cost function with respect to Z.</returns>
        private MatrixVectors ReLuPrime(MatrixVectors dA, MatrixVectors Z)
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

        /// <summary>
        ///     Initializes the weights and biases and returns them as a dictionary
        ///     the string key represents the name of the parameter as "W[l]" or "b[l]".
        /// </summary>
        /// <param name="dims">Number of neurons in each layer of the network.</param>
        /// <returns>Dictionary containing weights and bias'.</returns>
        public Dictionary<string, MatrixVectors> InitalizeParameters(int[] dims)
        {
            
            Dictionary<string, MatrixVectors> theta = new Dictionary<string, MatrixVectors>();
            for (int l = 1; l < dims.Length; l++)
            {
                MatrixVectors weights = new MatrixVectors(dims[l], dims[l - 1]);
                weights = weights.BroadcastScalar((float)Math.Sqrt(1 / dims[l - 1]), Operation.Multiply);
                MatrixVectors bias = new MatrixVectors(dims[l], 1);

                weights.InitializeRandom();

                theta.Add("W" + l, weights);
                theta.Add("b" + l, bias);
            }
            return theta;
        }

        /// <summary>
        ///     This method runs the linear function z = MatrixMultiplication(w, A_prev) + b.
        /// </summary>
        /// <param name="previousLayersActivations">A vector containing the previous layers activations.</param>
        /// <param name="weights">A matrix containing the weights.</param>
        /// <param name="bias">A vector containing the bias'.</param>
        /// <returns>
        ///     The linear cache which holds the weights, bias and the previous layers activations. Also returns Z.
        /// </returns>
        private Tuple<LinearCache, MatrixVectors> LinearForward(MatrixVectors previousLayersActivations, MatrixVectors weights, MatrixVectors bias)
        {
            MatrixVectors z = weights.Dot(previousLayersActivations).MatrixElementWise(bias, Operation.Add);
            LinearCache linearCache = new LinearCache(weights, bias, previousLayersActivations);

            return new Tuple<LinearCache, MatrixVectors>(linearCache, z);
        }

        /// <summary>
        ///     This method runs the linear function and the specified activation function
        ///     to calculate the Z and A of the current layer.
        /// </summary>
        /// <param name="previousLayersActivations">Vector of the previous layer's activations.</param>
        /// <param name="weights">Matrix of the current layers weights.</param>
        /// <param name="bias">Vector of the current layers bias'.</param>
        /// <param name="activation">The type of activation function to use.</param>
        /// <returns>
        ///     It returns a tuple with the cache as the first item and the final activations as
        ///     the second item.
        /// </returns>
        private Tuple<LinearCache, MatrixVectors, MatrixVectors> ActivationsForward(MatrixVectors previousLayersActivations, MatrixVectors weights, MatrixVectors bias, Activation activation)
        {
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

        /// <summary>
        ///     This methods job is the calculate the activations of each layer.
        ///     It uses input layer as the first layers previous activations
        ///     and uses theta to calculate the linear function for the activations.
        /// 
        ///     This method gathers the linear and z caches of every layer.
        ///     It will then generate a prediction(AL) as the final layers activations.
        /// </summary>
        /// <param name="xInput">The input layer of the network.</param>
        /// <param name="theta">The weights and biases of the network.</param>
        /// <param name="dims">Number of neurons in each layer of the network.</param>
        /// <returns>A tuple containing the linear and z caches along with the prediction.</returns>
        public Tuple<List<LinearCache>, List<MatrixVectors>, MatrixVectors> ForwardPropagation(MatrixVectors xInput, Dictionary<string, MatrixVectors> theta, int[] dims) 
        {   
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

        /// <summary>
        ///     This method uses the cross entropy cost function to caculate the losses.
        /// </summary>
        /// <param name="AL">The final prediction of the network ranging from 0 to 1.</param>
        /// <param name="_y">The true label 0 or 1.</param>
        /// <param name="lambda">The L2 regularization hyper-parameter.</param>
        /// <param name="theta">Dictionary containing weights and bias'.</param>
        /// <param name="dims">Number of neurons in each layer of the network.</param>
        /// <returns>A float value which is the calculated loss as well as its derivative.</returns>
        public float ComputeCost(MatrixVectors AL, MatrixVectors _y, float lambda, Dictionary<string, MatrixVectors> theta, int[] dims)
        {
            
            if (AL.columns > 1 || _y.columns > 1 || !AL.CompareShape(_y))
            {
                Console.WriteLine("Invalid YShape");
                return 0f;
            }

            float crossEntropyCost = 0;
            float regularizedCost = 0;

            for(int l = 1; l < dims.Length; l++)
            {
                regularizedCost += MatrixCalculations.MatrixSummation(MatrixCalculations.Square(theta["W" + l]));
            }

            regularizedCost *= lambda / 2;

            for(int y = 0; y < _y.rows; y++)
            {
                float currentAL = AL.MatrixVector[0, y];
                float currentY = _y.MatrixVector[0, y];
                float currentCost = (float)-(currentY * Math.Log10(currentAL) + (1 - currentY) * Math.Log10(1 - currentAL));
                crossEntropyCost += currentCost;
            }

            float totalCost = crossEntropyCost + regularizedCost;

            return totalCost;
        }

        /// <summary>
        ///     This method calculates the derivatives of the parameters and the 
        ///     derivative of the previous layers activations all with respect to to the
        ///     cross entropy cost function.
        /// </summary>
        /// <param name="dZ">The derivative of the cost function with respect to Z.</param>
        /// <param name="linearCache">A linear cache obtained from forward prop.</param>
        /// <param name="lambda">The L2 regularization hyper-parameter.</param>
        /// <returns>
        ///     The derivatives for gradient descent.
        /// </returns>
        private Tuple<MatrixVectors, MatrixVectors, MatrixVectors> LinearBackward(MatrixVectors dZ, LinearCache linearCache, float lambda)
        {

            MatrixVectors regularizedWeight = linearCache.weights.BroadcastScalar(lambda, Operation.Multiply);
            MatrixVectors dW = dZ.Dot(linearCache.previousLayersActivations.Transpose());
            MatrixVectors dWRegularized = dW.MatrixElementWise(regularizedWeight, Operation.Add);
            MatrixVectors db = dZ.MatrixAxisSummation(1);
            MatrixVectors dAPrev = linearCache.weights.Transpose().Dot(dZ);

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
            return new Tuple<MatrixVectors, MatrixVectors, MatrixVectors>(dWRegularized, db, dAPrev);
        }

        /// <summary>
        ///     This method will calculate dC with respect to Z from one of the specified activations
        ///     then use this dC/dZ to calculate the other derivatives.
        /// </summary>
        /// <param name="dA">The derivative of the cost function with respect to the activations.</param>
        /// <param name="Z">The linear function of the weights biases and previous layers activations.</param>
        /// <param name="linearCache">A linear cache obtained from forward prop.</param>
        /// <param name="activation">The type of activation to use. Corrosponds with the activation that was used for this layer during forward prop.</param>
        /// <param name="lambda">The L2 regularization hyper-parameter.</param>
        /// <returns>The derivatives provided from the <see cref="LinearBackward"/> function.</returns>
        private Tuple<MatrixVectors, MatrixVectors, MatrixVectors> ActivationsBackward(MatrixVectors dA, MatrixVectors Z, LinearCache linearCache, Activation activation, float lambda)
        {
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

        /// <summary>
        ///     This method calculates all the derivatives for each layer in the neural network.
        ///     It starts by calculating dC/dAL.
        ///     After this it will go through each layer starting from the last
        ///     calculating the derivative of the cost function with respect to 
        ///     W, b, and A of the previous layer.
        /// 
        ///     dW and db will be used for updating the parameters.
        /// 
        ///     dAPrev is passed to the next step of the back prop as it is used to calculate dZ at that step.
        /// </summary>
        /// <param name="Y">The true labels of the input data.</param>
        /// <param name="AL">The final predictions of the network.</param>
        /// <param name="linearCache">A linear cache obtained from forward prop.</param>
        /// <param name="zCache">A cache containing every computer linear function Z.</param>
        /// <param name="lambda">The L2 regularization hyper-parameter.</param>
        /// <returns>The derivatives to run gradient descent with.</returns>
        public Dictionary<string, MatrixVectors> BackwardPropagation(MatrixVectors Y, MatrixVectors AL, List<LinearCache> linearCache, List<MatrixVectors> zCache, float lambda)
        {
            Dictionary<string, MatrixVectors> gradients = new Dictionary<string, MatrixVectors>();
            List<LinearCache> linearCaches = linearCache;
            List<MatrixVectors> Zs = zCache;
            int layersCount = linearCaches.Count;

            MatrixVectors YDividedByAL = Y.MatrixElementWise(AL, Operation.Divide);
            MatrixVectors OneMinusY = Y.BroadcastScalar(1, Operation.Subtract, true);
            MatrixVectors OneMinusAL = AL.BroadcastScalar(1, Operation.Subtract, true);
            MatrixVectors OneMinusYDividedByOneMinusAL = OneMinusY.MatrixElementWise(OneMinusAL, Operation.Divide);

            MatrixVectors dAL_P1 = YDividedByAL.MatrixElementWise(OneMinusYDividedByOneMinusAL, Operation.Subtract);
            MatrixVectors dAL = dAL_P1.BroadcastScalar(-1, Operation.Multiply);

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

        /// <summary>
        ///     Uses gradient descent to update the weights and biases.
        /// </summary>
        /// <param name="theta">Dictionary containing the weights and bias' of the network.</param>
        /// <param name="gradients">The derivatives used to calculate gradient descent</param>
        /// <param name="dims">Number of neurons in each layer of the network.</param>
        /// <param name="alpha">The learning rate</param>
        /// <returns>The updated parameters theta.</returns>
        public Dictionary<string, MatrixVectors> UpdateParameters(Dictionary<string, MatrixVectors> theta, Dictionary<string, MatrixVectors> gradients, int[] dims, float alpha)
        {
            for(int l = 1; l < dims.Length; l++)
            {
                MatrixVectors dWxLearningRate = gradients["dW" + l].BroadcastScalar(alpha, Operation.Multiply);
                MatrixVectors dbxLearningRate = gradients["db" + l].BroadcastScalar(alpha, Operation.Multiply);
                theta["W" + l] = theta["W" + l].MatrixElementWise(dWxLearningRate, Operation.Subtract);
                theta["b" + l] = theta["b" + l].MatrixElementWise(dbxLearningRate, Operation.Subtract);
            }
            return theta;
        }

        /// <summary>
        ///     Predicts the true labels of the given input vectors.
        /// </summary>
        /// <param name="inputs">A list of input vectors.</param>
        /// <param name="trueLabels">A list of true labels 0 or 1.</param>
        /// <param name="theta">Dictionary containing the weights and bias' of the network.</param>
        /// <param name="dims">Number of neurons in each layer of the network.</param>
        public void Predict(List<MatrixVectors> inputs, List<int> trueLabels, Dictionary<string, MatrixVectors> theta, int[] dims)
        {
            /*When predicting we round the values. If it is greater then 0.5
             * round to 1 else round to 0.
             * return the exact prediction of the network as well as printing the
             * rounded prediction.
             */

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
