using System;

class Perceptron
{
    static void Main()
    {
        double[] T = { 1, 1, 1,
                       0, 1, 0,
                       0, 1, 0 };

        double[] H = { 1, 0, 1,
                       1, 1, 1,
                       1, 0, 1 };

        double[][] entradas = { T, H };
        double[] saidasDesejadas = { +1, -1 };

        double taxaAprendizado = 0.2;
        int maxEpocas = 20;

        double[] pesos = InicializarPesos();
        double bias = 1.0;

        Treinar(entradas, saidasDesejadas, ref pesos, ref bias, taxaAprendizado, maxEpocas);

        Console.WriteLine("\n Teste Inicial");
        TestarPerceptron("T", T, pesos, bias);
        TestarPerceptron("H", H, pesos, bias);

        Console.WriteLine("\n Teste Distorcido");

        double[] T_distorcido = { 1, 1, 1,
                                  0, 1, 0,
                                  0, 0, 0 };

        double[] H_distorcido = { 1, 0, 1,
                                  1, 1, 1,
                                  1, 1, 1 };

        TestarPerceptron("T_distorcido", T_distorcido, pesos, bias);
        TestarPerceptron("H_distorcido", H_distorcido, pesos, bias);
    }

    static double[] InicializarPesos()
    {
        double[] pesos = {0.5, 0.5, 0.4,
                          0.5, 0.5, 0.4,
                          0.5, 0.5, 0.4};
        return pesos;
    }

    static double CalcularSaida(double[] x, double[] pesos, double bias)
    {
        double soma = bias;
        for (int i = 0; i < x.Length; i++)
            soma += pesos[i] * x[i];
        return soma >= 0 ? +1 : -1;
    }

    static void AjustarPesos(double[] x, double d, double y, ref double[] pesos, ref double bias, double taxa)
    {
        double erro = d - y;
        for (int i = 0; i < x.Length; i++)
            pesos[i] += taxa * erro * x[i];
        bias += taxa * erro;
    }

    static void Treinar(double[][] entradas, double[] saidasDesejadas,
                        ref double[] pesos, ref double bias,
                        double taxa, int maxEpocas)
    {
        for (int epoca = 1; epoca <= maxEpocas; epoca++)
        {
            int erros = 0;

            for (int p = 0; p < entradas.Length; p++)
            {
                double y = CalcularSaida(entradas[p], pesos, bias);
                double d = saidasDesejadas[p];

                if (y != d)
                {
                    AjustarPesos(entradas[p], d, y, ref pesos, ref bias, taxa);
                    erros++;
                }
            }

            Console.WriteLine($"Época {epoca}: erros = {erros}");

            if (erros == 0)
            {
                Console.WriteLine("Treinamento concluído sem erros!");
                break;
            }
        }
    }

    static void TestarPerceptron(string nome, double[] entrada, double[] pesos, double bias)
    {
        double saida = CalcularSaida(entrada, pesos, bias);
        Console.WriteLine($"Entrada {nome} → Saída: {saida}");
    }
}
