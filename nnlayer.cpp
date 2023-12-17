// neuralnetw.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include "neuralnetw.h"

NNLayer::NNLayer(int _layerno, int _n, int _m, int _pqnt, double _rand_epsilon,
	NNLayer* _prev, NNLayer* _next, int _tracebufcnt, char* _tracebuffer, int* _tracebufw)
{
	layerno = _layerno;
	n = _n;
	m = _m;
	pqnt = _pqnt;
	rand_epsilon = _rand_epsilon;
	prev = _prev;
	next = _next;
	tracebufcnt = _tracebufcnt;
	Tracebuffer = _tracebuffer;
	tracebufw = _tracebufw;

	Theta = new double[n * (m + 1)];
	ThetaT = new double[n * (m + 1)];
	ThetaM = new double[n * m];
	ThetaMT = new double[n * m];
	Line = new dpoint[n];
	LineT = new dpoint[m + 1];
	A = new double[n];
	Z = new double[n];
	In = new double[m + 1];
	Delta = new double[n];
	DeltaB = new double[n * (m + 1)];
	Grad = new double[n * (m + 1)];
	Gsum = new double[n * (m + 1)];

	for (int i = 0; i < n; i++)
	{
		Line[i] = Theta + i * (m + 1);
		A[i] = 0;
		Z[i] = 0;
		Delta[i] = 0;
	}

	double* p1 = DeltaB;
	double* p2 = Grad;
	for (int i = 0; i < n * (m + 1); i++)
	{
		*p1++ = 0;
		*p2++ = 0;
	}

	for (int i = 0; i < m + 1; i++)
	{
		LineT[i] = ThetaT + i * n;
	}

	TInit();
}

NNLayer::~NNLayer()
{
	char mess[100];

	for (int i = 0; i < 100; i++)
	{
		mess[i] = 0;
	}

	try
	{
		delete[] DeltaB;
		delete[] Delta;
		delete[] Z;
		delete[] A;
		delete[] In;
		delete[] LineT;
		delete[] Line;
		delete[] ThetaMT;
		delete[] ThetaM;
		delete[] ThetaT;
		delete[] Theta;
		delete[] Grad;
		delete[] Gsum;
	}
	catch (const std::exception& e)
	{
		sprintf_s(mess, "Деструктор NNLayer: %s\n", e.what());
		WriteToTraceBuffer(tracebufcnt, Tracebuffer, tracebufw, mess);
	}
}

void NNLayer::ThetaTran()
{
	double* src = Theta;
	double* dest = ThetaT;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m + 1; j++)
		{
			*(dest + j * n) = *src;
			src++;
		}
		dest++;
	}	
	
	src = ThetaM;
	dest = ThetaMT;

	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < m; j++)
		{
			*(dest + j * n) = *src;
			src++;
		}
		dest++;
	}
}

void NNLayer::InitIter()
{
	for (int i = 0; i < n; i++)
	{
		A[i] = 0;
		Z[i] = 0;
	}

	for (int i = 0; i < n; i++)
	{

		Delta[i] = 0;
	}

	double* p1 = DeltaB;
	double* p2 = Grad;
	for (int i = 0; i < n * (m + 1); i++)
	{
		*p1++ = 0;
		*p2++ = 0;
	}
}

void NNLayer::Forward(double* X, double* Y)
{
	//double* pc = Y;
	double* x = X;
	double* y = Y;
	double* a = A;
	double* z = Z;
	//double* in = In;

	*y = 0.0;
	fill8arr(n, y);
	*z = 0.0;
	fill8arr(n, z);
	movsdq(m, In, x);

	/*for (int i = 0; i < n; i++)
	{
		*pc++ = 0;
	}*/

	/*for (int i = 0; i < m; i++)
	{
		*in++ = *x++;
	}

	x = X;*/
	/*for (int i = 0; i < n; i++)
	{
		*z++ = 0;
	}*/

	//z = Z;

	unsigned long long nm = 0;

	*(unsigned int*)&nm = m;
	*(((unsigned int*)&nm) + 1) = n;

	//double* th = Theta;
	forwbs(nm, Z, X, Theta);

	//for (int i = 0; i < n; i++)
	//{
		//double* th = Theta + i * (m + 1);
		//x = X;

		//*z += *th++; //Умноженное на 1.0
		//for (int j = 1; j < m + 1; j++)
		//{
			//*z += (*th++ * *x++);
		//}
		//v8mult(m, x, th, z);
		//th += m;
		//*a++ = Sigmoid(*z);
		//z++;
	//}

	/*a = A;
	z = Z;*/
	for (int i = 0; i < n; i++)
	{
     	*a++ = 1.0 / (1.0 + exp(-*z++));
	}

	movsdq(n, Y, A);
}

void NNLayer::BackDelta(double* Y)
{
	double* d = Delta;
	double* a = A;
	double* y = Y;

	for (int i = 0; i < n; i++)
	{
		*d++ = (*a++ - *y++);
	}
}

void NNLayer::BackDelta()
{
	double* d = Delta;
	double* dnext;
	double* a = A;
	double* th;

	th = this->next->ThetaT;
	int qnt = this->next->n;
	th += qnt;
	double* delta_next = this->next->Delta;

	for (int i = 0; i < n; i++)
	{
		*d = 0;
		dnext = delta_next;
		v8mult(qnt, dnext, th, d);
		th += qnt;

		/*for (int j = 0; j < this->next->n; j++)
		{
			*d += (*th++ * *dnext++);
		}*/

		*d *= (*a * (1 - *a));
		a++;
		d++;
	}
}

void NNLayer::BackGrad()
{
	/*double* dnext = this->Delta;
	double* a = A;
	double* g = DeltaB;
	double* gs;
	double* grad = Grad;
	double* t = Theta;*/
	unsigned long long nm = 0;

	*(unsigned int*)&nm = n;
	*(((unsigned int*)&nm) + 1) = m;

	v8gsum(nm, Gsum, In, Delta);

	/*gs = Gsum;
	for (int i = 0; i < n; i++)
	{
		double* x = In;
		*gs++ = 1.0 * *dnext;
		for (int j = 1; j < m + 1; j++)
		{
			*gs++ = *x++ * *dnext;
		}

		dnext++;
	}*/

	v8add(n * (m + 1), DeltaB, Gsum);

	/*gs = Gsum;
	for (int i = 0; i < n * (m + 1); i++)
	{
		*g++ += *gs++;
	}*/

	//g = DeltaB;

	movsdq(n * (m + 1), Grad, DeltaB);

	/*for (int i = 0; i < n * (m + 1); i++)
	{
		*grad++ = *g++;
	}*/
}

void NNLayer::UpdateWeights(double lambda, double lambdareg)
{
	double* grad = Grad;
	double* t = Theta;
	double* tm = ThetaM;

	for (int i = 0; i < (m + 1) * n; i++)
	{
		*t++ -= (*t * lambdareg / pqnt + (lambda * *grad));
		grad++;
	}

	t = Theta;
	for (int i = 0; i < n; i++)
	{
		t++;
		for (int j = 0; j < m; j++)
		{
			*tm++ = *t++;
		}
	}

	ThetaTran();
}

double NNLayer::Sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double NNLayer::SigmoidP(double x)
{
	double c = exp(-x);

	return c / ((1.0 + c) * (1.0 + c));
}

void NNLayer::TInit()
{
	double* p = Theta;
	double* tm = ThetaM;

	for (int i = 0; i < n * (m + 1); i++)
	{
		*p = (((double)rand()) / RAND_MAX) * 2 * rand_epsilon - rand_epsilon;
		p++;
	}

	p = Theta;
	for (int i = 0; i < n; i++)
	{
		p++;
		for (int j = 0; j < m; j++)
		{
			*tm++ = *p++;
		}
	}

	ThetaTran();
}

double NNLayer::Norma2(double* a, double* b, int qnt)
{
	double ret = 0;
	double* p1 = a;
	double* p2 = b;

	for (int i = 0; i < qnt; i++)
	{
		ret += *p1++ * *p2++;
	}

	return ret;
}

double NNLayer::Norma2(double* a, int qnt)
{
	double ret = 0;
	double* p = a;

	for (int i = 0; i < qnt; i++)
	{
		ret += *p * *p;
		p++;
	}

	return ret;
}

double NNLayer::Norma(double* a, int qnt)
{
	return sqrt(Norma2(a, qnt));
}

bool NNLayer::isFirst()
{
	return prev == NULL;
}

bool NNLayer::isLast()
{
	return next == NULL;
}
