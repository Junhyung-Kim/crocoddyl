///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/costs/state.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/kinostate.hpp"
#include "crocoddyl/multibody/residuals/kinostate1.hpp"
namespace crocoddyl {
namespace python {

void exposeResidualState() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelState> >();

  bp::class_<ResidualModelState, bp::bases<ResidualModelAbstract> >(
      "ResidualModelState",
      "This cost function defines a residual vector as r = x - xref, with x and xref as the current and reference\n"
      "state, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelState::calc, bp::args("self", "data", "x", "u"),
          "Compute the state cost.\n\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelState::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualModelState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .add_property("reference",
                    bp::make_function(&ResidualModelState::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelState::set_reference, "reference state");
}

}  // namespace python
}  // namespace crocoddyl

namespace crocoddyl {
namespace python {

void exposeResidualFlyState() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualFlyState> >();

  bp::class_<ResidualFlyState, bp::bases<ResidualModelAbstract> >(
      "ResidualFlyState",
      "This cost function defines a residual vector as r = x - xref, with x and xref as the current and reference\n"
      "state, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualFlyState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualFlyState::calc, bp::args("self", "data", "x", "u"),
          "Compute the state cost.\n\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualFlyState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualFlyState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualFlyState::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualFlyState::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .add_property("reference",
                    bp::make_function(&ResidualFlyState::get_reference, bp::return_internal_reference<>()),
                    &ResidualFlyState::set_reference, "reference state");
}

}  // namespace python
}  // namespace crocoddyl

namespace crocoddyl {
namespace python {

void exposeResidualFlyState1() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualFlyState1> >();

  bp::class_<ResidualFlyState1, bp::bases<ResidualModelAbstract> >(
      "ResidualFlyState1",
      "This cost function defines a residual vector as r = x - xref, with x and xref as the current and reference\n"
      "state, respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd, std::size_t>(
          bp::args("self", "state", "xref", "nu"),
          "Initialize the state cost model.\n\n"
          ":param state: state description\n"
          ":param xref: reference state (default state.zero())\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, Eigen::VectorXd>(
          bp::args("self", "state", "xref"),
          "Initialize the state cost model.\n\n"
          "The default nu value is obtained from state.nv.\n"
          ":param state: state description\n"
          ":param xref: reference state"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(
          bp::args("self", "state", "nu"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero().\n"
          ":param state: state description\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state cost model.\n\n"
          "The default reference state is obtained from state.zero(), and nu from state.nv.\n"
          ":param state: state description"))
      .def<void (ResidualFlyState1::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualFlyState1::calc, bp::args("self", "data", "x", "u"),
          "Compute the state cost.\n\n"
          ":param data: cost data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualFlyState1::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualFlyState1::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualFlyState1::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the state cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (ResidualFlyState1::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                        const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .add_property("reference",
                    bp::make_function(&ResidualFlyState1::get_reference, bp::return_internal_reference<>()),
                    &ResidualFlyState1::set_reference, "reference state");
}

}  // namespace python
}  // namespace crocoddyl
