all: test1 test2

test1 : main.o quicksort.o
	gcc -g -o test main.o quicksort.o -lpthread
	./test quicksort
	rm test

test2 : main.o quicksort.o
	gcc -g -o test main.o quicksort.o -lpthread
	./test quicksort_th_dy
	rm test

quicksort.o : quicksort.c
	gcc -g -c -o quicksort.o quicksort.c -lpthread

mergesort.o : mergesort.c
	gcc -g -c -o mergesort.o mergesort.c -lpthread

bucketsort.o : bucketsort.c
	gcc -g -c -o bucketsort.o bucketsort.c -lpthread

main.o : main.c
	gcc -g -c -o main.o main.c -lpthread

clean:
	rm *.o