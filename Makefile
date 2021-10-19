all: test1 test2 test3

test1 : test
	./test quicksort
	rm test

test2 : test
	./test quicksort_th
	rm test

test3 : test
	./test mergesort
	rm test

test4 : test
	./test mergesort_th
	rm test

test5 : test
	./test bucketsort
	rm test *.o

test: main.o quicksort.o mergesort.o bucketsort.o
	gcc -g -o test main.o quicksort.o mergesort.o bucketsort.o -lpthread

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