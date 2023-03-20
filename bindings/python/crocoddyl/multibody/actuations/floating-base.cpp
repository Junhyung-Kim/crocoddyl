///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/kino-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationKinoBase() {
  bp::register_ptr_to_python<boost::shared_ptr<crocoddyl::ActuationModelKinoBase> >();

  bp::class_<ActuationModelKinoBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelKinoBase",
      "Floating-base actuation models.\n\n"
      "It considers the first joint, defined in the Pinocchio model, as the floating-base joints.\n"
      "Then, this joint (that might have various DoFs) is unactuated.",
      bp::init<boost::shared_ptr<StateKinodynamic> >(bp::args("self", "state"),
                                                   "Initialize the floating-base actuation model.\n\n"
                                                   ":param state: state of multibody system"))
      .def("calc", &ActuationModelKinoBase::calc, bp::args("self", "data", "x", "u"),
           "Compute the floating-base actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the floating-base actuation model.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def("calcDiff", &ActuationModelKinoBase::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the floating-base actuation model.\n\n"
           "It computes the partial derivatives of the floating-base actuation. It assumes that calc\n"
           "has been run first. The reason is that the derivatives are constant and\n"
           "defined in createData. The derivatives are constant, so we don't write again these values.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def("createData", &ActuationModelKinoBase::createData, bp::args("self"),
           "Create the floating-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl

namespace crocoddyl {
namespace python {

void exposeActuationFloatingBase() {
  bp::register_ptr_to_python<boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> >();

  bp::class_<ActuationModelFloatingBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFloatingBase",
      "Floating-base actuation models.\n\n"
      "It considers the first joint, defined in the Pinocchio model, as the floating-base joints.\n"
      "Then, this joint (that might have various DoFs) is unactuated.",
      bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                   "Initialize the floating-base actuation model.\n\n"
                                                   ":param state: state of multibody system"))
      .def("calc", &ActuationModelFloatingBase::calc, bp::args("self", "data", "x", "u"),
           "Compute the floating-base actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the floating-base actuation model.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def("calcDiff", &ActuationModelFloatingBase::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the floating-base actuation model.\n\n"
           "It computes the partial derivatives of the floating-base actuation. It assumes that calc\n"
           "has been run first. The reason is that the derivatives are constant and\n"
           "defined in createData. The derivatives are constant, so we don't write again these values.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def("createData", &ActuationModelFloatingBase::createData, bp::args("self"),
           "Create the floating-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl

