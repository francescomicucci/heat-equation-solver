PROGRAM jimdouglas

   USE decomp_2d
   USE decomp_2d_constants
   USE mpi



   IMPLICIT NONE
   INTEGER, PARAMETER :: nx = 100
   INTEGER, PARAMETER :: nt = 100
   INTEGER,DIMENSION(2) :: coords_proc
   REAL(mytype), PARAMETER :: T = 1.0 ! time domain
   REAL(mytype), PARAMETER :: X = 1.0 ! space domain
   REAL(mytype) :: d_x , dt, timestep = 0 ! intervals and current timestep
   REAL(mytype) :: mat_coeff, dt_dx2 ! matrix A coefficient
   REAL(mytype) :: error,start, finish
   REAL(mytype) :: w = 0 ! real coefficient for thomas algorithm
   !	 a - sub-diagonal (means it is the diagonal below the main diagonal)
   !	 b - the main diagonal
   !	 c - sup-diagonal (means it is the diagonal above the main diagonal)
   !	 d - right part -> RHS
   !	 x - the answer
   !	 n - number of equations
   REAL(mytype),DIMENSION(:), ALLOCATABLE :: a,b,c
   REAL(mytype),DIMENSION(:,:,:), ALLOCATABLE :: dx,dy,dz,told,real_solution
   REAL(mytype),DIMENSION(:,:), ALLOCATABLE :: toldn, tolde, toldw, tolds, btoldew,btoldns
   !REAL(mytype),DIMENSION(:), ALLOCATABLE :: f

   INTEGER process_Rank, size_Of_Cluster, ierror
   INTEGER :: i,j,k,dim,P_row,P_col, dest_proc_ew, source_proc_ew, dest_proc_ns, source_proc_ns
   INTEGER    STATUS(MPI_STATUS_SIZE)
   INTEGER RTYPE

   CALL MPI_INIT(ierror)
   CALL MPI_COMM_SIZE(MPI_COMM_WORLD, size_Of_Cluster, ierror)
   CALL MPI_Comm_rank(MPI_COMM_WORLD, process_Rank, ierror)

   !Count number of processors per side
   P_col=FLOOR(SQRT(dble(size_Of_Cluster)))
   DO WHILE (MOD(size_Of_Cluster,P_col) .NE. 0)
      P_col=P_col-1
   END DO
   P_row=size_Of_Cluster/P_col

   CALL decomp_2d_init(nx,nx,nx,P_row,P_col)
   IF (process_Rank .EQ. 0) WRITE (*,*) P_row, P_col
   !Calculate coordinate of the processor in the cart
   CALL MPI_CART_COORDS(DECOMP_2D_COMM_CART_Z,process_Rank,2,coords_proc,ierror)
   !Find processor east and west
   CALL MPI_CART_SHIFT(DECOMP_2D_COMM_CART_Z, 0, 1, source_proc_ew, dest_proc_ew, ierror)
   !Find processor north and south
   CALL MPI_CART_SHIFT(DECOMP_2D_COMM_CART_Z, 1, 1, source_proc_ns, dest_proc_ns, ierror)


   !  WRITE (*,*) process_Rank, coords_proc, source_proc_ew, dest_proc_ew, source_proc_ns, dest_proc_ns
   IF (mytype==4) THEN
      RTYPE = MPI_REAL
    ELSE
      RTYPE = MPI_DOUBLE_PRECISION
    END IF  

   d_x = X / ( 1 + nx)
   dt = T / nt
   dt_dx2 = dt / (d_x**2)
   mat_coeff =  dt_dx2 / 2

   dim = nx**3

   ALLOCATE(a(zsize(3))); ALLOCATE(b(zsize(3))); ALLOCATE(c(zsize(3)));

   ALLOCATE(told(zstart(1):zend(1),zstart(2):zend(2),zstart(3):zend(3)));

   ALLOCATE(dx(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)));
   ALLOCATE(dy(ystart(1):yend(1),ystart(2):yend(2),ystart(3):yend(3)));
   ALLOCATE(dz(zstart(1):zend(1),zstart(2):zend(2),zstart(3):zend(3)));

   ALLOCATE(tolde(zstart(2):zend(2),zstart(3):zend(3)))
   ALLOCATE(toldw(zstart(2):zend(2),zstart(3):zend(3)))
   ALLOCATE(btoldew(zstart(2):zend(2),zstart(3):zend(3)))
   ALLOCATE(toldn(zstart(1):zend(1),zstart(3):zend(3)))
   ALLOCATE(tolds(zstart(1):zend(1),zstart(3):zend(3)))
   ALLOCATE(btoldns(zstart(1):zend(1),zstart(3):zend(3)))

   ALLOCATE(real_solution(0:nx+1,0:nx+1,0:nx+1));

   !Write(*,*) "Process ", process_Rank, " has ", source_proc_ew, " as source_proc_ew and ", dest_proc_ew, " as dest_proc_ew", source_proc_ns, " as source_proc_ns and ", dest_proc_ns, " as dest_proc_ns"

   a = - mat_coeff ! sub-diagonal
   b = 1 + 2 * mat_coeff ! diagonal
   c = - mat_coeff ! upper-diagonal

   ! w0_ijk = T (x_i, y_j, z_k, t= 0) = 0 for each i,j,k

   told = 0
   tolde = 0
   tolds = 0
   toldn = 0
   toldw = 0
   btoldew = 0
   btoldns = 0

   !Calculate real solution time independent
   do k = 0, nx+1
      do j = 0, nx+1
         do i = 0, nx+1
            real_solution(i,j,k) = exact_sol(i,j,k)
         end do
      end do
   end do

   CALL CPU_TIME(start)

   timestep = timestep + dt
   DO WHILE(timestep .LE. T)
      ! initializing rhs
      DO k = zstart(3), zend(3)
         DO j = zstart(2),zend(2)
            DO i = zstart(1),zend(1)
               !Forcing term
               dz(i,j,k) = dt * forcing_term(i,j,k,timestep + dt/2) + dt_dx2*(-6 * told(i,j,k))
               ! i-1
               IF(i .NE. 1) THEN
                  IF(i .NE. zstart(1)) THEN
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i-1,j,k)) !nb
                  ELSE
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(tolde(j,k)) !nb
                  END IF
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i-1,j,k)*sin(timestep)) ! boundary condition
               END IF
               ! i+1
               IF(i .NE. nx) THEN
                  IF(i .NE. zend(1)) THEN
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i+1,j,k)) !nb
                  ELSE
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(toldw(j,k)) !nb
                  END IF
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i+1,j,k)*sin(timestep)) ! boundary condition
               END IF
               ! j-1
               IF (j .NE. 1) THEN
                  IF(j .NE. zstart(2)) THEN
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i,j-1,k)) !nb
                  ELSE
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(toldn(i,k)) !nb
                  END IF
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i,j-1,k)*sin(timestep)) ! boundary condition
               END IF
               ! j+1
               IF (j .NE. nx) THEN
                  IF(j .NE. zend(2)) THEN
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i,j+1,k)) !nb
                  ELSE
                     dz(i,j,k) = dz(i,j,k) + dt_dx2*(tolds(i,k)) !nb
                  END IF
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i,j+1,k)*sin(timestep)) ! boundary condition

               END IF
               ! k-1
               IF (k .NE. 1) THEN
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i,j,k-1))
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i,j,k-1)*sin(timestep)) ! boundary condition
               END IF
               ! k+1
               IF (k .NE. nx) THEN
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(told(i,j,k+1))
               ELSE
                  dz(i,j,k) = dz(i,j,k) + dt_dx2*(real_solution(i,j,k+1)*sin(timestep)) ! boundary condition
               END IF

            END DO

         END DO
      END DO

      !THOMAS ON Z
      CALL threed_thomas_z(a,b,c,dz)
      CALL transpose_z_to_y(dz, dy)

      !THOMAS ON Y
      CALL threed_thomas_y(a,b,c,dy)
      CALL transpose_y_to_x(dy, dx)

      ! THOMAS ON X
      CALL threed_thomas_x(a,b,c,dx)
      CALL transpose_x_to_y(dx, dy) !NOT OPTIMAL
      CALL transpose_y_to_z(dy, dz)

      told = dz + told

      ! Send to west, receive from east
      IF (dest_proc_ew >= 0) THEN
         btoldew = told(zend(1),zstart(2):zend(2),zstart(3):zend(3))
         CALL MPI_SEND(btoldew, size(btoldew,1)*size(btoldew,2), RTYPE, dest_proc_ew, dest_proc_ew, DECOMP_2D_COMM_CART_Z,ierror)
      END IF
      IF (source_proc_ew >= 0) THEN
         CALL MPI_RECV(btoldew, size(btoldew,1)*size(btoldew,2), RTYPE, source_proc_ew, process_Rank, DECOMP_2D_COMM_CART_Z,status,ierror)
         tolde = btoldew
      END IF

      ! Send to east, receive from west
      IF (source_proc_ew >= 0) THEN
         btoldew = told(zstart(1),zstart(2):zend(2),zstart(3):zend(3))
         CALL MPI_SEND(btoldew, size(btoldew,1)*size(btoldew,2), RTYPE, source_proc_ew, source_proc_ew, DECOMP_2D_COMM_CART_Z,ierror)
      END IF
      IF (dest_proc_ew >= 0) THEN
         CALL MPI_RECV(btoldew, size(btoldew,1)*size(btoldew,2), RTYPE, dest_proc_ew, process_Rank, DECOMP_2D_COMM_CART_Z,status,ierror)
         toldw = btoldew
      END IF

      ! Send to south, receive from north
      IF (dest_proc_ns >= 0) THEN
         btoldns = told(zstart(1):zend(1),zend(2),zstart(3):zend(3))
         CALL MPI_SEND(btoldns, size(btoldns,1)*size(btoldns,2), RTYPE, dest_proc_ns, dest_proc_ns, DECOMP_2D_COMM_CART_Z,ierror)
      END IF
      IF (source_proc_ns >= 0) THEN
         CALL MPI_RECV(btoldns, size(btoldns,1)*size(btoldns,2), RTYPE, source_proc_ns, process_Rank, DECOMP_2D_COMM_CART_Z,status,ierror)
         toldn = btoldns
      END IF

      ! Send to north, receive from south
      IF (source_proc_ns >= 0) THEN
         btoldns = told(zstart(1):zend(1),zstart(2),zstart(3):zend(3))
         CALL MPI_SEND(btoldns, size(btoldns,1)*size(btoldns,2), RTYPE, source_proc_ns, source_proc_ns, DECOMP_2D_COMM_CART_Z,ierror)
      END IF
      IF (dest_proc_ns >= 0) THEN
         CALL MPI_RECV(btoldns, size(btoldns,1)*size(btoldns,2), RTYPE, dest_proc_ns, process_Rank, DECOMP_2D_COMM_CART_Z,status,ierror)
         tolds = btoldns
      END IF

      timestep = timestep + dt

   END DO

   CALL CPU_TIME(finish)


   PRINT '("Time = ",f15.10," seconds.")',(finish-start)

   real_solution = real_solution*sin(1.d0)

   error = NORM2(told - real_solution(zstart(1):zend(1),zstart(2):zend(2),zstart(3):zend(3)))**2

   !Print error for every processor
   WRITE(*,*) "Processor:", process_Rank, "with error= ",SQRT(error)/dim

   IF (process_Rank == 0) THEN
      CALL MPI_REDUCE(MPI_IN_PLACE, error, 1, RTYPE, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
      WRITE(*,*) "Error= ",SQRT(error)/dim
   ELSE
      CALL MPI_REDUCE(error, error, 1, RTYPE, MPI_SUM, 0, MPI_COMM_WORLD, ierror)
   END IF

   CALL decomp_2d_finalize()
   CALL MPI_FINALIZE(ierror)

CONTAINS

   SUBROUTINE threed_thomas_z(a,b,c,d)
      IMPLICIT NONE
      REAL(mytype),DIMENSION(:) :: a,b,c
      REAL(mytype),DIMENSION(zstart(1):zend(1),zstart(2):zend(2),zstart(3):zend(3)) :: d

      !Boundary conditions on z
      d(:,:,nx) = d(:,:,nx) + mat_coeff * real_solution(zstart(1):zend(1),zstart(2):zend(2),nx+1)*(sin(timestep+dt) - sin(timestep))

      DO k = 2, nx
         w = a(k) / b(k-1)
         b(k) = (1 + 2 * mat_coeff) - w * c(k-1)
         d(:,:,k) = d(:,:,k) - w * d(:,:,k-1)

      END DO

      d(:,:,nx) = d(:,:,nx) / b(nx)
      DO k=nx-1,1,-1
         d(:,:,k) = (d(:,:,k) - c(k) * d(:,:,k+1)) / b(k)
      END DO

   END SUBROUTINE threed_thomas_z

   SUBROUTINE threed_thomas_y(a,b,c,d)
      IMPLICIT NONE
      REAL(mytype),DIMENSION(:) :: a,b,c
      REAL(mytype),DIMENSION(ystart(1):yend(1),ystart(2):yend(2),ystart(3):yend(3)) :: d

      DO j = ystart(3),yend(3)

         !Boundary conditions on y
         d(:,nx,j) = d(:,nx,j) + mat_coeff * real_solution(ystart(1):yend(1),nx+1,j)*(sin(timestep+dt) - sin(timestep))

         DO k = 2, nx
            w = a(k) / b(k-1)
            b(k) = (1 + 2 * mat_coeff) - w * c(k-1)
            d(:,k,j) = d(:,k,j) - w * d(:,k-1,j)

         END DO

         d(:,nx,j) = d(:,nx,j) / b(nx)
         DO k=nx-1,1,-1
            d(:,k,j) = (d(:,k,j) - c(k) * d(:,k+1,j)) / b(k)
         END DO
      END DO
   END SUBROUTINE threed_thomas_y

   SUBROUTINE threed_thomas_x(a,b,c,d)
      IMPLICIT NONE
      REAL(mytype),DIMENSION(:) :: a,b,c
      REAL(mytype),DIMENSION(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) :: d
      INTEGER :: i, j

      DO i = xstart(3),xend(3)
         DO j = xstart(2),xend(2)

            !Boundary conditions on x
            d(nx,j,i) = d(nx,j,i) + mat_coeff * real_solution(nx+1,j,i)*(sin(timestep+dt) - sin(timestep))

            DO k = 2, nx
               w = a(k) / b(k-1)
               b(k) = (1 + 2 * mat_coeff) - w * c(k-1)
               d(k,j,i) = d(k,j,i) - w * d(k-1,j,i)

            END DO

            d(nx,j,i) = d(nx,j,i) / b(nx)
            DO k=nx-1,1,-1
               d(k,j,i) = (d(k,j,i) - c(k) * d(k+1,j,i)) / b(k)
            END DO
         END DO
      END DO
   END SUBROUTINE threed_thomas_x

   FUNCTION exact_sol(i,j,k) RESULT (exact_s)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: i,j,k
      REAL(mytype) :: exact_s
      exact_s = sin(i*d_x) * sin(j*d_x) * sin(k*d_x)

   END FUNCTION exact_sol

   DOUBLEPRECISION FUNCTION forcing_term(i,j,k,t)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: i,j,k
      REAL(mytype), INTENT(IN) :: t

      forcing_term = (cos(t) + 3*sin(t))*sin(i*d_x)*sin(j*d_x)*sin(k*d_x)

   END FUNCTION forcing_term

END PROGRAM
