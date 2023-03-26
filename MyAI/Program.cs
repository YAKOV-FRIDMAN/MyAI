using System;
using System.Linq;

namespace MyAi
{


public class Program
{
    public static void Main(string[] args)
    {
        // צור רשת נוירונית עם 3 שכבות: שכבת קלט עם 2 נוירונים, שכבה מוסתרת עם 3 נוירונים ושכבת פלט עם 1 נוירון
        NeuralNetwork nn = new NeuralNetwork(2, 3, 1);

        // נתוני למידה - הקלט (XOR function)
        float[][] inputs = new float[][]
        {
            new float[] { 0, 0 },
            new float[] { 0, 1 },
            new float[] { 1, 0 },
            new float[] { 1, 1 }
        };

        // נתוני למידה - פלט צפוי
        float[][] expectedOutputs = new float[][]
        {
            new float[] { 0 },
            new float[] { 1 },
            new float[] { 1 },
            new float[] { 0 }
        };

        // אמן את הרשת הנוירונית על נתוני הלמידה
        int epochs = 5000; // מספר העברות על נתוני הלמידה
        float learningRate = 0.1f; // שיעור הלמידה
        for (int i = 0; i < epochs; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                nn.Train(inputs[j], expectedOutputs[j], learningRate);
            }
        }

        // בדוק את ביצועי הרשת הנוירונית על קלט חדש
        float[] newInput = new float[] { 0, 1 };
        float[] output = nn.FeedForward(newInput);

        Console.WriteLine($"Input: [{newInput[0]}, {newInput[1]}], Output: {output[0]}");

    }
}


/// <summary>
/// בטח! רשת נוירונים מלאכותית (ANN) היא מודל מתמקד בלמידה של מחשב, המדמה את פעולות המערכת העצבית של יצורים חיים. הרעיון הוא ליצור מערכת יכולה ללמוד מדוגמאות ולבצע פעולות שונות, כמו זיהוי תמונות, תרגום שפות, ניתוח טקסט וכו'.
/// רשת נוירונים מורכבת משכבות של נוירונים, כולל שכבת קלט, שכבות מוסתרות ושכבת פלט. כל נוירון מחובר לנוירונים בשכבה הבאה באמצעות משקלות. במהלך הלמידה, הרשת מעדכנת את המשקלות בהתאם לשגיאות שהיא מצטברת.אז ניתן לך דוגמה פשוטה של רשת נוירונים עם שכבה קלט, שכבה מוסתרת ושכבת פלט בשפת C#:
/// </summary>
public class NeuralNetwork
{
    /// <summary>
    /// שכבות
    /// </summary>
    private int[] layers;
    /// <summary>
    /// נוירונים
    /// </summary>
    private float[][] neurons;
    /// <summary>
    /// משקולות
    /// </summary>
    private float[][][] weights;
    /// <summary>
    /// הטיות
    /// </summary>
    private float[][] biases;
    /// <summary>
    /// 
    /// </summary>
    /// <param name="layers">שכבות</param>
    public NeuralNetwork(params int[] layers)
    {
        this.layers = layers;
        InitializeNeurons();
        InitializeWeights();
        InitializeBiases();
    }
    /// <summary>
    /// אתחול נוירונים
    /// </summary>
    private void InitializeNeurons()
    {
        neurons = new float[layers.Length][];
        for (int i = 0; i < layers.Length; i++)
        {
            neurons[i] = new float[layers[i]];
        }
    }
    /// <summary>
    /// אתחול משקלים
    /// </summary>
    private void InitializeWeights()
    {
        weights = new float[layers.Length - 1][][];
        for (int i = 0; i < layers.Length - 1; i++)
        {
            weights[i] = new float[layers[i + 1]][];
            for (int j = 0; j < layers[i + 1]; j++)
            {
                weights[i][j] = new float[layers[i]];
            }
        }
    }
    /// <summary>
    /// אתחול הטיות
    /// </summary>
    private void InitializeBiases()
    {
        biases = new float[layers.Length - 1][];
        for (int i = 0; i < layers.Length - 1; i++)
        {
            biases[i] = new float[layers[i + 1
                ]];
        }
    }
    /// <summary>
    /// הזנה קדימה
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public float[] FeedForward(float[] inputs)
    {
        Array.Copy(inputs, neurons[0], inputs.Length);

        for (int i = 1; i < layers.Length; i++)
        {
            for (int j = 0; j < neurons[i].Length; j++)
            {
                float value = 0f;
                for (int k = 0; k < neurons[i - 1].Length; k++)
                {
                    value += weights[i - 1][j][k] * neurons[i - 1][k];
                }
                neurons[i][j] = Sigmoid(value + biases[i - 1][j]);
            }
        }

        return neurons[neurons.Length - 1];
    }
    /// <summary>
    /// פונקציית סיגמואיד היא פונקציה מתמטית, בעלת עקומה בצורת האות "S" הנקראת גם עקומת סיגמואיד. לעיתים קרובות "פונקציית סיגמואיד"
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private float Sigmoid(float x)
    {
        return (float)(1 / (1 + Math.Exp(-x)));
    }
    /// <summary>
    ///  פונקציית האיפור החזרתי לקוד הקיים של הרשת הנוירונית. פונקציית הלמידה מקבלת את הקלט, הפלט הצפוי ושיעור הלמידה, ומעדכנת את המשקלות והעיוותים על בסיס השגיאות שהרשת מצטברת.
    /// </summary>
    /// <param name="inputs"></param>
    /// <param name="expectedOutputs"></param>
    /// <param name="learningRate"></param>
    public void Train(float[] inputs, float[] expectedOutputs, float learningRate)
    {
        //הזן קדימה כדי לקבל את הפלט של הרשת
        // Feed forward to get the network's output
        float[] networkOutput = FeedForward(inputs);
        //חשב שגיאות פלט
        // Calculate output errors
        float[][] outputErrors = new float[layers.Length][];
        outputErrors[layers.Length - 1] = new float[layers[layers.Length - 1]];
        for (int i = 0; i < layers[layers.Length - 1]; i++)
        {
            outputErrors[layers.Length - 1][i] = (expectedOutputs[i] - networkOutput[i]) * SigmoidDerivative(networkOutput[i]);
        }
        //חשב שגיאות עבור שכבות נסתרות
        // Calculate errors for hidden layers
        for (int i = layers.Length - 2; i > 0; i--)
        {
            outputErrors[i] = new float[layers[i]];
            for (int j = 0; j < layers[i]; j++)
            {
                float sum = 0f;
                for (int k = 0; k < layers[i + 1]; k++)
                {
                    sum += outputErrors[i + 1][k] * weights[i][k][j];
                }
                outputErrors[i][j] = sum * SigmoidDerivative(neurons[i][j]);
            }
        }
        // דכון משקלים והטיות 
        // Update weights and biases
        for (int i = 0; i < layers.Length - 1; i++)
        {
            for (int j = 0; j < layers[i + 1]; j++)
            {
                for (int k = 0; k < layers[i]; k++)
                {
                    weights[i][j][k] += learningRate * outputErrors[i + 1][j] * neurons[i][k];
                }
                biases[i][j] += learningRate * outputErrors[i + 1][j];
            }
        }
    }
    /// <summary>
    /// נגזרת סיגמואידית
    /// </summary>
    /// <param name="x"></param>
    /// <returns></returns>
    private float SigmoidDerivative(float x)
    {
        float sigmoid = Sigmoid(x);
        return sigmoid * (1 - sigmoid);
    }
}
}