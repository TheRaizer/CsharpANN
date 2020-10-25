using System;

namespace ANN
{
    public class MatrixVectors
    {
        public float[,] MatrixVector { get; set; }
        public readonly int rows;
        public readonly int columns;

        public MatrixVectors(int rows, int columns)
        {
            MatrixVector = new float[columns, rows];
            this.columns = columns;
            this.rows = rows;
        }

        public Tuple<int, int> Shape()
        {
            return new Tuple<int, int>(rows, columns);
        }

        public void InitializeRandom()
        {
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    float val = (float)(Program.rand.NextDouble() * (0.05 + 0.05) - 0.05);
                    MatrixVector[x, y] = val;
                }
            }
        }

        public void InitializeWithZeros()
        {
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    MatrixVector[x, y] = 0;
                }
            }
        }

        public void InputValuesIntoMatrix()
        {
            for(int y = 0; y < rows; y++)
            {
                string[] nums = Console.ReadLine().Split(' ');
                for(int x = 0; x < columns; x++)
                {
                    MatrixVector[x, y] = float.Parse(nums[x]);
                }
            }
        }

        public void OutputMatrixValue()
        {
            Console.WriteLine("");
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    Console.Write(MatrixVector[x, y] + " ");
                }
                Console.WriteLine("");
            }
        }
    }
}
