using System;

namespace ANN
{
    public static class MatrixCalculations
    {
        public static MatrixVectors Dot(MatrixVectors matrix_1, MatrixVectors matrix_2)
        {
            if (matrix_1.columns != matrix_2.rows)//the number of columns in the first matrix must be equal to the number of rows in the second
            {
                Console.WriteLine("Error in Matrix multiplication");
                return null;
            }

            MatrixVectors outputMatrix = new MatrixVectors(matrix_1.rows, matrix_2.columns);//the output matrix uses the columns of the second matrix and the rows of the first

            for(int c = 0; c < matrix_2.columns; c++)
            {
                for(int y = 0; y < matrix_1.rows; y++)
                {
                    float value = 0;
                    for(int x = 0; x < matrix_1.columns; x++)
                    {
                        value += matrix_1.MatrixVector[x, y] * matrix_2.MatrixVector[c, x];
                    }

                    outputMatrix.MatrixVector[c, y] = value;
                }
            }
            return outputMatrix;
        }

        public static MatrixVectors MatrixElementWise(MatrixVectors matrix_1, MatrixVectors matrix_2, Operation operation)
        {
            if(matrix_1.columns != matrix_2.columns || matrix_1.rows != matrix_2.rows)
            {
                Console.WriteLine("Cannot do Matrix element wise multiplications ");
                return null;
            }

            MatrixVectors outputMatrix = new MatrixVectors(matrix_1.rows, matrix_1.columns);

            for (int y = 0; y < outputMatrix.rows; y++)
            {
                for (int x = 0; x < outputMatrix.columns; x++)
                {
                    switch (operation)
                    {
                        case Operation.Add:
                            outputMatrix.MatrixVector[x, y] = matrix_1.MatrixVector[x, y] + matrix_2.MatrixVector[x, y];
                            break;
                        case Operation.Subtract:
                            outputMatrix.MatrixVector[x, y] = matrix_1.MatrixVector[x, y] - matrix_2.MatrixVector[x, y];
                            break;
                        case Operation.Multiply:
                            outputMatrix.MatrixVector[x, y] = matrix_1.MatrixVector[x, y] * matrix_2.MatrixVector[x, y];
                            break;
                        case Operation.Divide:
                            outputMatrix.MatrixVector[x, y] = matrix_1.MatrixVector[x, y] / matrix_2.MatrixVector[x, y];
                            break;
                    }
                }
            }
            return outputMatrix;
        }
        public static float MatrixSummation(MatrixVectors matrix)
        {
            float sum = 0;
            for (int y = 0; y < matrix.rows; y++)
            {
                for (int x = 0; x < matrix.columns; x++)
                {
                    sum += matrix.MatrixVector[x, y];
                }
            }
            return sum;
        }

        public static MatrixVectors MatrixAxisSummation(MatrixVectors matrix, int axis)
        {
            ///<summary>
            /// axis = 0 returns a row vector of all the rows summed together.
            /// axis = 1 returns a column vector of all the columns summed together.
            /// In numpy this is equivalent to np.sum with KeepDims always True.
            ///</summary>
            
            if(axis == 0)
            {
                MatrixVectors summedOverColumns = new MatrixVectors(1, matrix.columns);
                for (int x = 0; x < matrix.columns; x++)
                {
                    for (int y = 0; y < matrix.rows; y++)
                    {
                        summedOverColumns.MatrixVector[x, 0] += matrix.MatrixVector[x, y];
                    }
                }
                return summedOverColumns;
            }
            else if(axis == 1)
            {
                MatrixVectors summedOverRows = new MatrixVectors(matrix.rows, 1);
                for (int y = 0; y < matrix.rows; y++)
                {
                    for (int x = 0; x < matrix.columns; x++)
                    {
                        summedOverRows.MatrixVector[0, y] += matrix.MatrixVector[x, y];
                    }
                }
                return summedOverRows;
            }
            else
            {
                throw new ArgumentOutOfRangeException();
            }
        }

        public static MatrixVectors Exp(MatrixVectors matrix)
        {
            MatrixVectors outputMatrix = new MatrixVectors(matrix.rows, matrix.columns);
            for(int y = 0; y < matrix.rows; y++)
            {
                for(int x = 0; x < matrix.columns; x++)
                {
                    outputMatrix.MatrixVector[x, y] = (float)(Math.Pow(Math.E, matrix.MatrixVector[x, y]));
                }
            }

            return outputMatrix;
        }

        public static MatrixVectors BroadcastScalar(MatrixVectors matrix, float scalar, Operation operation, bool reverse = false)
        {
            MatrixVectors outputMatrix = new MatrixVectors(matrix.rows, matrix.columns);

            for(int y = 0; y < matrix.rows; y++)
            {
                for (int x = 0; x < matrix.columns; x++)
                {
                    switch (operation)
                    {
                        case Operation.Add:
                            outputMatrix.MatrixVector[x, y] = matrix.MatrixVector[x, y] + scalar;
                            break;
                        case Operation.Subtract:
                            if(reverse)
                                outputMatrix.MatrixVector[x, y] = scalar - matrix.MatrixVector[x, y];
                            else
                                outputMatrix.MatrixVector[x, y] = matrix.MatrixVector[x, y] - scalar;
                            break;
                        case Operation.Multiply:
                            outputMatrix.MatrixVector[x, y] = matrix.MatrixVector[x, y] * scalar;
                            break;
                        case Operation.Divide:
                            if(reverse)
                                outputMatrix.MatrixVector[x, y] = scalar / matrix.MatrixVector[x, y];
                            else
                                outputMatrix.MatrixVector[x, y] = matrix.MatrixVector[x, y] / scalar;
                            break;
                    }
                }
            }
            return outputMatrix;
        }
        public static MatrixVectors Maximum(MatrixVectors matrix_1, float scalar = 0, MatrixVectors matrix_2 = null)
        {
            if (matrix_2 != null)
            {
                if (!matrix_1.CompareShape(matrix_2))
                {
                    Console.WriteLine("Matrix shapes do not align");
                    return null;
                }
            }
            MatrixVectors maximizedMatrix = new MatrixVectors(matrix_1.rows, matrix_1.columns);
            for (int y = 0; y < maximizedMatrix.rows; y++)
            {
                for(int x = 0; x < maximizedMatrix.columns; x++)
                {
                    if (matrix_2 != null)
                    {
                        maximizedMatrix.MatrixVector[x, y] = Math.Max(matrix_1.MatrixVector[x, y], matrix_2.MatrixVector[x, y]);
                    }
                    else
                    {
                        maximizedMatrix.MatrixVector[x, y] = Math.Max(scalar, matrix_1.MatrixVector[x, y]);
                    }
                }
            }

            return maximizedMatrix;
        }

        public static MatrixVectors Transpose(MatrixVectors matrix)
        {
            MatrixVectors matrixTranspose = new MatrixVectors(matrix.columns, matrix.rows);

            for(int y = 0; y < matrixTranspose.rows; y++)
            {
                for(int x = 0; x < matrixTranspose.columns; x++)
                {
                    matrixTranspose.MatrixVector[x, y] = matrix.MatrixVector[y, x];
                }
            }

            return matrixTranspose;
        }
    }
}
