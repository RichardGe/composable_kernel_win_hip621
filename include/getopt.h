#ifndef GETOPT_H

#define GETOPT_H

extern int opterr;		/* if error message should be printed */
extern int optind;		/* index into parent argv vector */
extern int optopt;		/* character checked for validity */
extern int optreset;  	/* reset getopt  */
extern char *optarg;	/* argument associated with option */

// Define option struct, similar to POSIX `getopt`
struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};


int getopt(int nargc, char * const nargv[], const char *ostr);

int getopt_long(int argc, char *const argv[], const char *optstring,
                const struct option *longopts, int *longindex);
				
				
/*
enum    		
{
  no_argument = 0,      	
  required_argument,		
  optional_argument		
};
*/

#define  no_argument  0
#define  required_argument  1


#endif



