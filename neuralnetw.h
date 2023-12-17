//���������� ������������ ���������� ��� �������������� ����������
//�����: �.�. �����������
//������ 1.0
//17 ������� 2023 �.

#pragma once

#define _DLLAPI extern "C" __declspec(dllexport)

#include <cmath>
#include "stdio.h"
#include <codecvt>
#include <locale>
#include "assert.h"

extern "C" void movsdd(long qnt, void* dest, void* src);

extern "C" void movsdq(long qnt, void* dest, void* src);

//extern "C" void movsDD(int qnt, void* dest, void* src);

extern "C" void matrix8tran(int n, int m, void* src, void* dest);

extern "C" void fill8arr(long n, void* p); //���������� ������� p �� n 8-������� ��������� ������ �� 1 ��������

extern "C" void v8mult(long qnt, void* v1, void* v2, void* ret);

extern "C" void v8add(long qnt, void* v1, void* v2);

extern "C" void v8gsum(unsigned long long qnt, void* gsum, void* x, void* delta);

extern "C" void forwbs(unsigned long long qnt, void* z, void* x, void* theta);

extern "C" void Matrix8Tran(int n, int m, void* src, void* dest);

using dpoint = double *;

class NNDataPrepare
{
public:
	int qnt, n_in, n_out;

	double* X;
	double* cv;
	double gmin, gmax;

	NNDataPrepare(int _qnt, int _n_in, int _n_out, double* x);
	~NNDataPrepare();

	void Normalyze(double* in, double* out);

	void EvalCVmatrix(double* x);
};

class NNLayer
{
public:

	int layerno, n, m;   //����� ���� � ����������� ������� �����
	int pqnt;            //�-�� ��������� ��������
	double rand_epsilon;
	char* Tracebuffer;     //����� �����������
	int tracebufcnt;       //������ ������ �����������
	int* tracebufw;         //�������� � ����� �����������
	double* Theta;         //� ����������� �������� ��������� ��������
	double* ThetaM;        //��� ������������ ������� ��������� ��������
	double* ThetaT;
	double* ThetaMT;
	dpoint* Line;
	dpoint* LineT;
	double* A;
	double* Z;
	double* In;
	double* Delta;
	double* DeltaB;
	double* Grad;
	double* Gsum;        //��������������� ���������� ��� ���������� ���������
	NNLayer* prev;
	NNLayer* next;

	NNLayer(int _layerno, int _n, int _m, int _pqnt, double _rand_epsilon,
		NNLayer* _prev, NNLayer* _next, int _tracebufcnt, char* _tracebuffer, int* _tracebufw);

	~NNLayer();

	void ThetaTran();

	void Forward(double* X, double* Y);

	void BackDelta(double* Y);     //��� ���������� ����

	void BackDelta();              //��� �� ��������� �����

	void BackGrad();

	void InitIter();

	void UpdateWeights(double lambda = 1.0, double r = 0);

	double Sigmoid(double x);

	double SigmoidP(double x);

	void TInit();

	double Norma2(double* a, double* b, int qnt);

	double Norma2(double* a, int qnt);

	double Norma(double* a, int qnt);
	
	bool isFirst();

	bool isLast();
};

//��������� ���� ��� ������������ ����������
class NN
{
public:

	int in_n, out_n; //�-�� ������ � ������� ����
	int nlayers;     //�-�� �����
	int pqnt;        //�-�� ��������� ��������
	double rand_epsilon;   //��������� ��� ��������� ������������� ������ �����
	char* Tracebuffer;     //����� �����������
	int tracebufcnt;       //������ ������ �����������
	int *tracebufw;         //�������� � ����� �����������
	double J;        //������� ���������
	double* X;       //������ ����������� pqnt x in_n - ����� ���� ��� ������� ���������� �������
	double* Y;       //������ ����������� pqnt x out_n - ������ ���� ��� ������� ���������� �������
	double** LineX;  //������ ���������� �� ������ ������� X
	double** LineY;  //������ ���������� �� ������ ������� Y
	double* XL;
	double* YL;
	NNLayer* Wfirst;
	NNLayer* Wlast;
	int nli, nlo;

	NN(int n, int _nlayers, int _pqnt, double _rand_epsilon, int* nout, int _tracebufcnt, char* _tracebuffer, int* _tracebufw, int processorno);

	~NN();

	//�������� pqnt �������� ��� �������� ����
	void SetPatterns(double* _X, double* _Y);

	double Propagation(double lambda = 1.0, double epsilon = 0.1, double lambdareg = 0);

	double Learn(double lambda = 1.0, double epsilon = 0.1, double lambdareg = 0);

	void Recognize(double* _X, double* _Y);

	void UpdateWeights(double h, double r);
};

//�������� ����� ��������� ����. �������� ������� ����� � ���� ������������� ������
class NeuralNetwork
{
public:

	int in_n, out_n; //�-�� ������ � ������� ����
	int nlayers;     //�-�� �����
	int nthreads;    //�-�� �������
	int pqnt;        //�-�� ��������� ��������
	double rand_epsilon;   //��������� ��� ��������� ������������� ������ �����
	char* Tracebuffer;     //����� �����������
	int tracebufcnt;       //������ ������ �����������
	int tracebufw;         //�������� � ����� �����������
	double J;        //������� ���������
	double* X;       //������ ����������� pqnt x in_n - ����� ���� ��� ������� ���������� �������
	double* Y;       //������ ����������� pqnt x out_n - ������ ���� ��� ������� ���������� �������
	NNLayer* Wfirst;
	NNLayer* Wlast;
	int nli, nlo;
	NN** parnetw;    //������ ���������� �� ��������� ���� ��� ������������ ����������

	NeuralNetwork(int n, int _nlayers, int _pqnt, int _nthreads, double _rand_epsilon, int* nout, int _tracebufcnt, char* _tracebuffer);

	NeuralNetwork(char* fname, int tbcnt, char* buf);

	~NeuralNetwork();

	//�������� pqnt �������� ��� �������� ����
	void SetPatterns(double* _X, double* _Y);

	void InpRand();

	double Learn(int maxiter = 100, double lambda = 1.0, double epsilon = 0.1, double lambdareg = 0);

	void Recognize(double* _X, double* _Y);

	int Save(wchar_t* fname);   //���������� ��������� ��������� ����

	double UpdateWeights(double h, double r);

	void ZeroGrad(); //��������� ��������� �� ���� �����

	void GetParallelGrad();

	void SetParallelTheta();

	double NormaIter();

	void UpdateLayersWeights(double h, double r);
};

double LearnParallel(NN* netw, double lambda, double epsilon, double lambdareg);

//�������������� �������
_DLLAPI long long _stdcall nnCreate(int n, int _nlayers, int _pqnt, int _nthreads, double _rand_epsilon, int* nout, int _tracebufcnt = 0, char* _tracebuffer = NULL);

_DLLAPI long long _stdcall nnLoad(char* fname, int tbcnt, char* buf);

_DLLAPI void _stdcall nnDelete(long long nnet);

_DLLAPI void _stdcall nnSetPatterns(long long nnet, double* _X, double* _Y);

_DLLAPI double _stdcall nnLearn(long long nnet, int maxiter = 100, double lambda = 1.0, double epsilon = 0.1, double lambdareg = 0);

_DLLAPI void _stdcall nnRecognize(long long nnet, double* _X, double* _Y);

_DLLAPI int _stdcall nnSave(long long nnet, wchar_t* fname);

_DLLAPI int _stdcall nnGetOutQnt(long long nnet);

_DLLAPI long long _stdcall nndpCreate(int _qnt, int _n_in, int _n_out, double* x);

_DLLAPI int _stdcall nndpGetQnt(long long dp);

_DLLAPI void _stdcall nndpGetX(long long dp, double* x);

_DLLAPI double _stdcall nndpGetMax(long long dp);

_DLLAPI double _stdcall nndpGetMin(long long dp);

_DLLAPI void _stdcall nndpNormalyze(long long dp, double* in, double* out);

_DLLAPI void _stdcall nndpDelete(long long dp);

