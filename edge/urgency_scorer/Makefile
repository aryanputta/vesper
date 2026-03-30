CC     = gcc
CFLAGS = -O2 -Wall -fPIC -shared
TARGET = urgency_scorer.so
SRC    = urgency.c

all: $(TARGET)

$(TARGET): $(SRC) urgency.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

clean:
	rm -f $(TARGET) urgency_test

test: urgency_test
	./urgency_test

urgency_test: urgency.c urgency.h
	$(CC) -O2 -Wall -DTEST_MAIN -o urgency_test urgency.c -lm
