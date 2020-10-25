using System;

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
            ANNTest();
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

            Console.Write("X1: ");
            int columns_1 = int.Parse(Console.ReadLine());//columns is equal to the x value of a grid
            Console.Write("Y1: ");
            int rows_1 = int.Parse(Console.ReadLine());//rows is equal to the y value of the grid

            Console.Write("X2: ");
            int columns_2 = int.Parse(Console.ReadLine());
            Console.Write("Y2: ");
            int rows_2 = int.Parse(Console.ReadLine());

            MatrixVectors matrix_1 = new MatrixVectors(rows_1, columns_1);
            matrix_1.InputValuesIntoMatrix();
            MatrixVectors matrix_2 = new MatrixVectors(rows_2, columns_2);
            matrix_2.InputValuesIntoMatrix();

            if (matrix_calc == "dot")
            {
                MatrixVectors matrix = MatrixCalculations.MatrixMultiplication(matrix_1, matrix_2);
                matrix.OutputMatrixValue();
            }
            else if (matrix_calc == "element")
            {
                MatrixVectors matrix = MatrixCalculations.MatrixElementWise(matrix_1, matrix_2, Operation.Multiply);
                matrix.OutputMatrixValue();
            }
            else if (matrix_calc == "scalar add")
            {
                MatrixVectors matrix = MatrixCalculations.BroadcastScalar(matrix_1, 2, Operation.Add);
                matrix.OutputMatrixValue();
            }
        }
    }
}
