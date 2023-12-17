_text SEGMENT

public movsdq
public movsdd
public movsdw
public movsdb
public matrix8tran

; опирование области пам€ти по адресу R8 в область пам€ти по адресу rdx по 8 байт в цикле из rcx повторений
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

; опирование области пам€ти по адресу R8 в область пам€ти по адресу rdx по 4 байта в цикле из rcx повторений
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

; опирование области пам€ти по адресу R8 в область пам€ти по адресу rdx по 2 байта в цикле из rcx повторений
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

; опирование области пам€ти по адресу R8 в область пам€ти по адресу rdx по 1 байту в цикле из rcx повторений
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

;«аполнение массива p из n 8-байтных элементов числом из 1 элемента
;ѕараметры: rcx - n, rdx - адрес массива
fill8arr proc
        push rdi

        mov rax, qword ptr [rdx]
		mov rdi, rdx
		cld
		rep stosq

		pop rdi

        ret
fill8arr endp

;¬ычисление сигмоиды от вектора из n 8-байтных элементов (double)
;ѕараметры: rcx - n, rdx - адрес вектора результата, r8 - адрес исходного вектора
nsigmoid  proc
        finit

		fld1
        fldz
        fsub  qword ptr [r8] 

        ;ƒописать алгоритм

        ret
nsigmoid  endp

;ќсновной расчет forward
;ѕараметры: rcx - по 4 байта n и m, rdx - адрес Z, r8 - адрес вектора X, r9 - адрес матрицы Theta
forwbs  proc
        push r10
		push rsi
		push rdi

        push rdx
		pop  rsi                       ;адрес Z
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

;¬ычисление градиента Gsum
;ѕараметры: rcx - по 4 байта n и m, rdx - адрес Gsum, r8 - адрес вектора X, r9 - адрес вектора Delta
v8gsum  proc
        push r10
		push rsi
		push rdi

        push rdx
		pop  rsi                       ;адрес Gsum
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

;—кал€рное произведение векторов с n 8-байтными компонентами (double)
;ѕараметры: rcx - n, rdx - адрес 1 вектора, r8 - адрес 2 вектора, r9 - адрес переменной, к которой добавл€етс€ результат
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

;—ложение векторов с n 8-байтными компонентами (double), результат - в 1 векторе
;ѕараметры: rcx - n, rdx - адрес 1 вектора, r8 - адрес 2 вектора
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

;“ранспонирование матрицы n * m чисел double
;ѕараметры: rcx - n, rdx - m, r8 - адрес исходной матрицы, r9 - адрес транспонированной матрицы
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
		shl r10, 3    ;–азмер строки в байтах транспонированной матрицы
		xor r11, r11  ;ѕеременна€ цикла i
	l1: mov r13, r9
	    mov r14, r11
		shl r14, 3
		add r13, r14
		xor r12, r12  ;ѕеременна€ цикла j
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
