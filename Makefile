all: quicksort

quicksort : main.o
	gcc -o quicksort_test main.o

sorts.o : sorts.c
	gcc -c -o sorts.o sorts.c

main.o : main.c
	gcc -c -o main.o main.c

clean:
	rm *.o main