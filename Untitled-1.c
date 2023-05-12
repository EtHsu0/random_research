#include <string.h>
#include <stdio.h>
char src[25] = "Hello!";
char dst[25];

int main(void) {
    strcpy(dst, src);
    printf("%s\n", src);
    printf("%s\n", dst);
    while(1);
}
