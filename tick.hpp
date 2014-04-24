#ifndef TICK_H
#define TICK_H

#if (defined(_WIN32) || defined(_WIN64))
#  include <windows.h>
#else
#  include <time.h>
#  include <sys/time.h>
#endif

static unsigned long
tick(void)
{
#if (defined(_WIN32) || defined(_WIN64))	
	return (unsigned long)GetTickCount();
#else
	unsigned long c;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	c = tv.tv_sec * 1000;
	c += tv.tv_usec / 1000;
	return c;
#endif
}

#endif
