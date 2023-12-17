_text SEGMENT

public movsdq
public movsdd
public movsdw
public movsdb
public matrix8tran

;����������� ������� ������ �� ������ R8 � ������� ������ �� ������ rdx �� 8 ���� � ����� �� rcx ����������
movsdq proc
        push rdi
		push rsi

        cld
		mov rdi, rdx
		mov rsi, r8
		rep movsq

		pop rsi
		pop rdi

        ret
movsdq endp

;����������� ������� ������ �� ������ R8 � ������� ������ �� ������ rdx �� 4 ����� � ����� �� rcx ����������
movsdd proc
        push rdi
		push rsi

        cld
		mov rdi, rdx
		mov rsi, r8
		rep movsd

		pop rsi
		pop rdi

        ret
movsdd endp

;����������� ������� ������ �� ������ R8 � ������� ������ �� ������ rdx �� 2 ����� � ����� �� rcx ����������
movsdw proc
        push rdi
		push rsi

        cld
		mov rdi, rdx
		mov rsi, r8
		rep movsw

		pop rsi
		pop rdi

        ret
movsdw endp

;����������� ������� ������ �� ������ R8 � ������� ������ �� ������ rdx �� 1 ����� � ����� �� rcx ����������
movsdb proc
        push rdi
		push rsi

        cld
		mov rdi, rdx
		mov rsi, r8
		rep movsb

		pop rsi
		pop rdi

        ret
movsdb endp

;���������� ������� p �� n 8-������� ��������� ������ �� 1 ��������
;���������: rcx - n, rdx - ����� �������
fill8arr proc
        push rdi

        mov rax, qword ptr [rdx]
		mov rdi, rdx
		cld
		rep stosq

		pop rdi

        ret
fill8arr endp

;���������� �������� �� ������� �� n 8-������� ��������� (double)
;���������: rcx - n, rdx - ����� ������� ����������, r8 - ����� ��������� �������
nsigmoid  proc
        finit

		fld1
        fldz
        fsub  qword ptr [r8] 

        ;�������� ��������

        ret
nsigmoid  endp

;�������� ������ forward
;���������: rcx - �� 4 ����� n � m, rdx - ����� Z, r8 - ����� ������� X, r9 - ����� ������� Theta
forwbs  proc
        push r10
		push rsi
		push rdi

        push rdx
		pop  rsi                       ;����� Z
		xor  rdx, rdx
		mov  edx, ecx            	   ;m
		shr  rcx, 32                   ;n
		mov  r10, rdx

        finit

l1:		fldz
        mov  rdi, r8
        fld  qword ptr [r9] 
		faddp st(1), st(0)
		add  r9, 8

l2:     fld qword ptr [rdi]
        add  rdi, 8
        fmul qword ptr [r9]
		add  r9, 8
		faddp st(1), st(0)
		dec  rdx
		jnz  l2

		mov  rdx, r10
		fstp qword ptr [rsi]
		add  rsi, 8
		dec  rcx
		jnz  l1

		pop rdi
		pop rsi
		pop r10
        ret
forwbs  endp

;���������� ��������� Gsum
;���������: rcx - �� 4 ����� n � m, rdx - ����� Gsum, r8 - ����� ������� X, r9 - ����� ������� Delta
v8gsum  proc
        push r10
		push rsi
		push rdi

        push rdx
		pop  rsi                       ;����� Gsum
		xor  rdx, rdx
		mov  edx, ecx            	   ;m
		shr  rcx, 32                   ;n
		xchg rcx, rdx
		mov  r10, rdx

        finit

l1:		mov  rdi, r8
		mov  rax, [r9]
		mov  [rsi], rax
		add  rsi, 8

l2:     fld qword ptr [rdi]
        add  rdi, 8
        fmul qword ptr [r9]
		fstp qword ptr [rsi]
		add  rsi, 8
		dec  rdx
		jnz  l2

		mov  rdx, r10
		add  r9, 8
		dec  rcx
		jnz  l1

		pop rdi
		pop rsi
		pop r10
        ret
v8gsum  endp

;��������� ������������ �������� � n 8-�������� ������������ (double)
;���������: rcx - n, rdx - ����� 1 �������, r8 - ����� 2 �������, r9 - ����� ����������, � ������� ����������� ���������
v8mult  proc

        finit
		fldz
l1:		or  rcx, rcx
		jz  _ret

        fld qword ptr [rdx]
        fmul qword ptr [r8]
		faddp st(1), st(0)
		dec rcx
		add rdx, 8
		add r8, 8
		jmp l1

_ret:
        ;fxch st(1)
        fadd qword ptr [r9]
		fstp qword ptr [r9]
 
        ret
v8mult  endp

;�������� �������� � n 8-�������� ������������ (double), ��������� - � 1 �������
;���������: rcx - n, rdx - ����� 1 �������, r8 - ����� 2 �������
v8add   proc

        finit
l1:		
        or  rcx, rcx
		jz  _ret

        fld qword ptr [rdx]
        fadd qword ptr [r8]
		fstp qword ptr [rdx]
		dec rcx
		add rdx, 8
		add r8, 8
		jmp l1

_ret:
        ret
v8add   endp

;���������������� ������� n * m ����� double
;���������: rcx - n, rdx - m, r8 - ����� �������� �������, r9 - ����� ����������������� �������
matrix8tran proc
        push r15
        push r14
        push r13
		push r12
		push r11
		push r10
        push r9
		push r8
		push rdx
		push rcx

        mov r10, rcx
		shl r10, 3    ;������ ������ � ������ ����������������� �������
		xor r11, r11  ;���������� ����� i
	l1: mov r13, r9
	    mov r14, r11
		shl r14, 3
		add r13, r14
		xor r12, r12  ;���������� ����� j
    l2: mov r15, [r8]
		mov [r13], r15
		add r13, r10
		add r8, 8
		inc r12
		cmp r12, rdx
		jb  l2
		inc r11
		cmp r11, rcx
		jb  l1

		pop rcx
		pop rdx
		pop r8
		pop r9
		pop r10
		pop r11
		pop r12
		pop r13
		pop r14
		pop r15

        ret
matrix8tran endp

_text ENDS

END
