using System;
using System.Collections.Generic;

namespace ANN
{
    public enum Activation
    {
       Sigmoid,
       ReLu,
       SoftMax
    }

    class Program
    {
        public readonly static Random rand = new Random();
        private static NeuralNetwork neuralNetwork = new NeuralNetwork();

        static void Main(string[] args)
        {
            //ANNTest();
            //InputTest();
            Console.Write("W_X1: ");
            int columns_1 = int.Parse(Console.ReadLine());//columns is equal to the x value of a grid
            Console.Write("W_Y1: ");
            int rows_1 = int.Parse(Console.ReadLine());//rows is equal to the y value of the grid
            Console.Write("axis: ");
            int axis = int.Parse(Console.ReadLine());

            MatrixVectors matrix = new MatrixVectors(rows_1, columns_1);
            matrix.InputValuesIntoMatrix();

            MatrixVectors matrixSummed = MatrixCalculations.MatrixAxisSummation(matrix, axis);
            matrixSummed.OutputMatrixValue();
        }

        private static void ANNTest()
        {
            int[] dims = { 3, 2, 1 };
            neuralNetwork.InitalizeParameters(dims);
            MatrixVectors inputVector = new MatrixVectors(dims[0], 1);

            for(int y = 0; y < inputVector.columns; y++)
            {
                inputVector.MatrixVector[0, y] = (float)(rand.NextDouble() * (0.1 + 0.1) - 0.1);
            }
            neuralNetwork.ForwardPropagation(inputVector, neuralNetwork.theta, dims);
        }

        private static void InputTest()
        {
            Console.Write("calculation type: ");
            string matrix_calc = Console.ReadLine();

            Console.Write("W_X1: ");
            int columns_1 = int.Parse(Console.ReadLine());//columns is equal to the x value of a grid
            Console.Write("W_Y1: ");
            int rows_1 = int.Parse(Console.ReadLine());//rows is equal to the y value of the grid

            Console.Write("X_X2: ");
            int columns_2 = int.Parse(Console.ReadLine());
            Console.Write("X_Y2: ");
            int rows_2 = int.Parse(Console.ReadLine());

            Console.Write("b_X2: ");
            int columns_3 = int.Parse(Console.ReadLine());
            Console.Write("b_Y2: ");
            int rows_3 = int.Parse(Console.ReadLine());

            MatrixVectors matrix_W = new MatrixVectors(rows_1, columns_1);
            matrix_W.InputValuesIntoMatrix();
            MatrixVectors vector_X = new MatrixVectors(rows_2, columns_2);
            vector_X.InputValuesIntoMatrix();
            MatrixVectors vector_b = new MatrixVectors(rows_3, columns_3);
            vector_b.InputValuesIntoMatrix();

            if (matrix_calc == "dot")
            {
                MatrixVectors matrix = MatrixCalculations.Dot(matrix_W, vector_X);
                matrix.OutputMatrixValue();
            }
            else if (matrix_calc == "element")
            {
                MatrixVectors matrix = MatrixCalculations.MatrixElementWise(matrix_W, vector_X, Operation.Multiply);
                matrix.OutputMatrixValue();
            }
            else if (matrix_calc == "scalar add")
            {
                MatrixVectors matrix = MatrixCalculations.BroadcastScalar(matrix_W, 2, Operation.Add);
                matrix.OutputMatrixValue();
            }
            else if(matrix_calc == "forward")
            {
                int[] dims = { 3, 2 };
                Dictionary<string, MatrixVectors> theta = new Dictionary<string, MatrixVectors>();
                theta.Add("W1", matrix_W);
                theta.Add("b1", vector_b);
                neuralNetwork.ForwardPropagation(vector_X, theta, dims);
            }
        }
    }
}
