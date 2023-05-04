
#include <mpi.h>

#include "gtest_mpi.hpp"

GTEST_API_ int main(int argc, char **argv) {
    int test_result;

    testing::InitGoogleTest(&argc, argv);

    int proc_rank;
    int proc_numb;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_numb);

    // on MPI ranks != 0 remove the default output listeners if there are any
    if( proc_rank != 0 )
    {
      ::testing::TestEventListener* defaultListener = ::testing::UnitTest::GetInstance()->listeners().default_result_printer();
      ::testing::UnitTest::GetInstance()->listeners().Release(defaultListener);
      delete defaultListener;

      ::testing::TestEventListener* defaultXMLListener = ::testing::UnitTest::GetInstance()->listeners().default_xml_generator();
      ::testing::UnitTest::GetInstance()->listeners().Release(defaultXMLListener);
      delete defaultXMLListener;
    }

    test_result=RUN_ALL_TESTS();

    MPI_Finalize();

    return test_result;
}
