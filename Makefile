all: test

test : main.o
	gcc -o test main.o

quicksort.o : quicksort.c
	gcc -c -o quicksort.o quicksort.c

mergesort.o : mergesort.c
	gcc -c -o mergesort.o mergesort.c

bucketsort.o : bucketsort.c
	gcc -c -o bucketsort.o bucketsort.c

main.o : main.c
	gcc -c -o main.o main.c

clean:
	rm *.o main