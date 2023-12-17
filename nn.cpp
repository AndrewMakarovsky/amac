// nn.cpp.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include "neuralnetw.h"

NN::NN(int n, int _nlayers, int _pqnt, double _rand_epsilon, int* nout, int _tracebufcnt, char* _tracebuffer, int* _tracebufw, int processorno)
{
	in_n = n;
	nlayers = _nlayers;
	pqnt = _pqnt;
	rand_epsilon = _rand_epsilon;
	Tracebuffer = _tracebuffer;
	tracebufcnt = _tracebufcnt;
	tracebufw = _tracebufw;
	J = 0;
	X = NULL;
	Y = NULL;
	Wfirst = NULL;
	nli = nlo = 0;

	int n1 = n;
	NNLayer* prev = NULL;
	for (int i = 1; i < nlayers; i++)
	{
		NNLayer* p = new NNLayer(i + 1, nout[i - 1], n1, pqnt, rand_epsilon, prev, NULL, tracebufcnt, Tracebuffer, tracebufw);
		if (prev != NULL)
		{
			prev->next = p;
			if (n1 > nli)
			{
				nli = n1;
			}
			if (nout[i - 1] > nlo)
			{
				nlo = nout[i - 1];
			}
		}
		else
		{
			Wfirst = p;
			nli = n1;
			nlo = nout[i - 1];
		}

		prev = p;
		n1 = nout[i - 1];
	}

	Wlast = prev;
	out_n = n1;

	X = new double[pqnt * in_n];
	Y = new double[pqnt * out_n];
	LineX = new dpoint[pqnt];
	LineY = new dpoint[pqnt];
	XL = new double[nli];
	YL = new double[nlo];

	for (int i = 0; i < pqnt; i++)
	{
		LineX[i] = (dpoint)(X + i * in_n);
		LineY[i] = (dpoint)(Y + i * out_n);
	}

	if (tracebufcnt - *tracebufw > 0)
	{
		char mess[100];

		for (int i = 0; i < 100; i++)
		{
			mess[i] = 0;
		}

		sprintf_s(mess, "%s %d %s\n", "Neural network for processor", processorno, "is created");

		int len = (int)strlen(mess);

		if (tracebufcnt - *tracebufw > len)
		{
			for (int i = 0; i < len; i++)
			{
				*(Tracebuffer + *tracebufw) = mess[i];
				(*tracebufw)++;
			}
		}
	}
}

NN::~NN()
{
	NNLayer* p = Wlast;
	NNLayer* prev = NULL;
	char mess[100];

	for (int i = 0; i < 100; i++)
	{
		mess[i] = 0;
	}

	try
	{
		delete[] X;
		delete[] Y;
		delete[] LineX;
		delete[] LineY;
		delete[] XL;
		delete[] YL;
	}
	catch (const std::exception& e)
	{
		sprintf_s(mess, "Деструктор NN точка 1: %s\n", e.what());
		WriteToTraceBuffer(tracebufcnt, Tracebuffer, tracebufw, mess);
	}

	try
	{
		for (int i = 1; i < nlayers; i++)
		{
			prev = p->prev;
			delete p;

			if (prev == NULL)
			{
				break;
			}
			else
			{
				p = prev;
			}
		}
	}
	catch (const std::exception& e)
	{
		sprintf_s(mess, "Деструктор NN точка 2: %s\n", e.what());
		WriteToTraceBuffer(tracebufcnt, Tracebuffer, tracebufw, mess);
	}
}

//Загрузка pqnt образцов для обучения сети
void NN::SetPatterns(double* _X, double* _Y)
{
	double* x = X;
	double* y = Y;
	double* _x = _X;
	double* _y = _Y;

	for (int i = 0; i < pqnt * in_n; i++)
	{
		*x++ = *_x++;
	}

	for (int i = 0; i < pqnt * out_n; i++)
	{
		*y++ = *_y++;
	}
}

double NN::Learn(double lambda, double epsilon, double lambdareg)
{
	double norma = 0, cnorma;
	NNLayer* cl = Wfirst;
	while (cl)
	{
		cl->InitIter();
		cl = cl->next;
	}

	J = 0;

	for (int t = 0; t < pqnt; t++)
	{
		NNLayer* clevel = Wfirst;

		//Forward propagation
		while (clevel)
		{
			if (clevel->isFirst())
			{
				double* p1 = XL;
				double* p2 = LineX[t];
				movsdq(in_n, p1, p2);
				/*for (int i = 0; i < in_n; i++)
				{
					*p1++ = *p2++;
				}*/
			}
			else
			{
				double* p1 = XL;
				double* p2 = YL;
				movsdq(clevel->m, p1, p2);
				/*for (int i = 0; i < clevel->m; i++)
				{
					*p1++ = *p2++;
				}*/
			}

			clevel->Forward(XL, YL);
			clevel = clevel->next;
		}

		double* yt = LineY[t];
		double* ht = Wlast->A;
		int hqnt = Wlast->n;
		//JV[t] = 0;

		for (int i = 0; i < hqnt; i++)
		{
			J -= (yt[i] * log(ht[i]) + (1 - yt[i]) * log(1 - ht[i]));
		}

		//Back propagation
		clevel = Wlast;
		clevel->BackDelta(LineY[t]);  //Для последнего слоя
		clevel = clevel->prev;

		while (clevel)
		{
			clevel->BackDelta();     //Для не последних слоев

			clevel = clevel->prev;
		}

		cnorma = 0;
		clevel = Wlast;
		for (int i = 0; i < clevel->n; i++)
		{
			cnorma += clevel->Delta[i] * clevel->Delta[i];
		}

		norma += cnorma;

		clevel = Wlast;
		while (clevel)
		{
			clevel->BackGrad();

			clevel = clevel->prev;
		}
	}

	if (lambdareg > 0)
	{
		double r = 0;
		int lnum = 0;
		NNLayer* l = Wfirst;

		while (l != NULL)
		{
			r += l->Norma2(l->ThetaM, l->n * l->m);
			l = l->next;
			lnum++;
		}

		r *= (lambdareg / lnum);
		J += r;
	}

	return norma;
}

void NN::UpdateWeights(double h, double r)
{
	NNLayer* cl = Wfirst;
	while (cl)
	{
		cl->UpdateWeights(h, r);
		cl = cl->next;
	}
}

void NN::Recognize(double* _X, double* _Y)
{
	NNLayer* clevel = Wfirst;
	while (clevel)
	{
		if (clevel->isFirst())
		{
			memmove((void*)XL, (void*)_X, in_n * sizeof(double));
		}
		else
		{
			memmove((void*)XL, (void*)YL, clevel->m * sizeof(double));
		}

		clevel->Forward(XL, YL);
		clevel = clevel->next;
	}

	memmove((void*)_Y, (void*)YL, Wlast->n * sizeof(double));
}
