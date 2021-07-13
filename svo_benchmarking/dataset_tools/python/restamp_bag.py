#!/usr/bin/env python

# ------------------------------------------------------------------------------
# Function : restamp ros bagfile (using header timestamps)
# Project  : IJRR MAV Datasets
# Author   : www.asl.ethz.ch
# Version  : V01  21JAN2016 Initial version.
# Comment  :
# Status   : under review
#
# Usage    : python restamp_bag.py -i inbag.bag -o outbag.bag
# ------------------------------------------------------------------------------

import roslib
import rosbag
import rospy
import sys
import getopt
from   std_msgs.msg import String

def main(argv):

    inputfile = ''
    outputfile = ''

    # parse arguments
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'usage: restamp_bag.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'usage: python restamp_bag.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg

    # print console header
    print ""
    print "restamp_bag"
    print ""
    print 'input file:  ', inputfile
    print 'output file: ', outputfile
    print ""
    print "starting restamping (may take a while)"
    print ""

    outbag = rosbag.Bag(outputfile, 'w')
    messageCounter = 0
    kPrintDotReductionFactor = 1000

    try:
        for topic, msg, t in rosbag.Bag(inputfile).read_messages():

            # Write message in output bag with input message header stamp
            outbag.write(topic, msg, msg.header.stamp)

            if (messageCounter % kPrintDotReductionFactor) == 0:
                    #print '.',
                    sys.stdout.write('.')
                    sys.stdout.flush()
            messageCounter = messageCounter + 1

    # print console footer
    finally:
        print ""
        print ""
        print "finished iterating through input bag"
        print "output bag written"
        print ""
        outbag.close()

if __name__ == "__main__":
   main(sys.argv[1:])

