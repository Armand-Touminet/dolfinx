// Copyright (C) 2004 Johan Jansson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2005.

#include <dolfin/dolfin_math.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/PETScManager.h>
#include <dolfin/NewVector.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewVector::NewVector() : x(0), copy(false)
{
  // Initialize PETSc
  PETScManager::init();
}
//-----------------------------------------------------------------------------
NewVector::NewVector(uint size) : x(0), copy(false)
{
  if(size < 0)
    dolfin_error("Size of vector must be non-negative.");

  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  init(size);
}
//-----------------------------------------------------------------------------
NewVector::NewVector(Vec x) : x(x), copy(true)
{
  // Initialize PETSc 
  PETScManager::init();
}
//-----------------------------------------------------------------------------
NewVector::NewVector(const Vector &x) : x(0), copy(false)
{
  // Initialize PETSc
  PETScManager::init();

  // Create PETSc vector
  init(x.size());

  const uint n = size();
  for (uint i = 0; i < n; i++)
    setvalue(i, x(i));
}
//-----------------------------------------------------------------------------
NewVector::~NewVector()
{
  clear();
}
//-----------------------------------------------------------------------------
void NewVector::init(uint size)
{
  // Two cases:
  //
  //   1. Already allocated and dimension changes -> reallocate
  //   2. Not allocated -> allocate
  //
  // Otherwise do nothing
  
  if (x)
  {
    const uint n = this->size();

    if (n == size)
    {
      return;      
    }
  }
  else
  {
    clear();
  }

  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, size);
  VecSetFromOptions(x);
}
//-----------------------------------------------------------------------------
void NewVector::axpy(const real a, const NewVector& x) const
{
  VecAXPY(&a, x.vec(), this->x);
}
//-----------------------------------------------------------------------------
void NewVector::add(const real block[], const int cols[], int n)
{
  VecSetValues(x, n, cols, block, ADD_VALUES); 
}
//-----------------------------------------------------------------------------
void NewVector::apply()
{
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
void NewVector::clear()
{
  if ( x && !copy )
  {
    VecDestroy(x);
  }

  x = 0;
}
//-----------------------------------------------------------------------------
dolfin::uint NewVector::size() const
{
  int n = 0;
  VecGetSize(x, &n);

  return static_cast<uint>(n);
}
//-----------------------------------------------------------------------------
Vec NewVector::vec()
{
  return x;
}
//-----------------------------------------------------------------------------
const Vec NewVector::vec() const
{
  return x;
}
//-----------------------------------------------------------------------------
real* NewVector::array()
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);

  return data;
}
//-----------------------------------------------------------------------------
const real* NewVector::array() const
{
  dolfin_assert(x);

  real* data = 0;
  VecGetArray(x, &data);

  return data;
}
//-----------------------------------------------------------------------------
void NewVector::restore(real data[])
{
  VecRestoreArray(x, &data);
}
//-----------------------------------------------------------------------------
void NewVector::restore(const real data[]) const
{
  // Cast away the constness and trust PETSc to do the right thing
  real* tmp = const_cast<real *>(data);
  VecRestoreArray(x, &tmp);
}
//-----------------------------------------------------------------------------
NewVector::Index NewVector::operator() (uint i)
{
  Index index(i, *this);

  return index;
}
//-----------------------------------------------------------------------------
const NewVector& NewVector::operator= (const NewVector& x)
{
  VecCopy(x.vec(), this->x);

  return *this;
}
//-----------------------------------------------------------------------------
const NewVector& NewVector::operator= (real a)
{
  VecSet(&a, x);

  return *this;
}
//-----------------------------------------------------------------------------
void NewVector::disp() const
{
  VecView(x, PETSC_VIEWER_STDOUT_SELF);
}
//-----------------------------------------------------------------------------
void NewVector::setvalue(uint i, const real r)
{
  VecSetValue(x, static_cast<int>(i), r, INSERT_VALUES);

  VecAssemblyBegin(x);
  VecAssemblyEnd(x);
}
//-----------------------------------------------------------------------------
real NewVector::getvalue(uint i) const
{
  // Assumes uniprocessor case.

  real val = 0.0;

  PetscScalar *array = 0;
  VecGetArray(x, &array);
  val = array[i];
  VecRestoreArray(x, &array);

  return val;
}
//-----------------------------------------------------------------------------
// NewVector::Index
//-----------------------------------------------------------------------------
NewVector::Index::Index(uint i, NewVector& x) : i(i), x(x)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void NewVector::Index::operator =(const real r)
{
  x.setvalue(i, r);
}
//-----------------------------------------------------------------------------
NewVector::Index::operator real() const
{
  return x.getvalue(i);
}
//-----------------------------------------------------------------------------
