///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actions/kinodyn.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl
{

  template <typename Scalar>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::DifferentialActionModelKinoDynamicsTpl(
      boost::shared_ptr<StateKinodynamic> state, boost::shared_ptr<ActuationModelAbstract> actuation,
      boost::shared_ptr<CostModelSum> costs)
      : Base(state, actuation->get_nu(), costs->get_nr()),
        actuation_(actuation),
        costs_(costs),
        pinocchio_(*state->get_pinocchio().get()),
        without_armature_(true),
        armature_(VectorXs::Zero(state->get_nv()))
  {
    if (costs_->get_nu() != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
    }
    VectorXs temp;
    temp.resize(actuation->get_nu() + 4);
    temp.setZero();
    temp.head(nu_) = pinocchio_.effortLimit.head(nu_);

    for (int i =0; i < nu_; i++)
    {
      temp(i) = 2.0;
    }

    temp(nu_) = 0.5;
    temp(nu_ + 1) = 5;
    temp(nu_ + 2) = 0.5;
    temp(nu_ + 3) = 5;
    Base::set_u_lb(Scalar(-1.) * temp);
    Base::set_u_ub(Scalar(+1.) * temp);
  }

  template <typename Scalar>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::~DifferentialActionModelKinoDynamicsTpl() {}

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
      const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }

    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());

    actuation_->calc(d->multibody.actuation, x, u);

    // Computing the dynamics using ABA or manually for armature case
    /* if (without_armature_) {
       d->xout = pinocchio::aba(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau.segment(0,state_->get_nv()));
       pinocchio::updateGlobalPlacements(pinocchio_, d->pinocchio);
     } else {

       pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
       d->pinocchio.M.diagonal() += armature_;
       pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
       d->Minv.setZero();
       pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);
       d->u_drift = d->multibody.actuation->tau - d->pinocchio.nle;
       d->xout.noalias() = d->Minv * d->u_drift;
     }*/

    d->xout = d->multibody.actuation->tau;
    d->xout = d->multibody.actuation->tau.segment(0, state_->get_nv());
    d->xout2 << x_state[1], 11.9411 * x_state[0] - 11.9411 * x_state[2] - d->multibody.actuation->u_x[1] * 1.0 / 78.8188, d->multibody.actuation->u_x[0], d->multibody.actuation->u_x[1], x_state[5], 11.9411 * x_state[4] - 11.9411 * x_state[6] + d->multibody.actuation->u_x[3] * 1.0 / 78.8188, d->multibody.actuation->u_x[2], d->multibody.actuation->u_x[3]; // d->dhg;

    // Computing the cost value and residuals
    costs_->calc(d->costs, x, u);
    d->cost = d->costs->cost;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calc(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    Data *d = static_cast<Data *>(data.get());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),state_->get_nv());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(4);

    costs_->calc(d->costs, x);
    d->cost = d->costs->cost;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
      const Eigen::Ref<const VectorXs> &u)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }

    const std::size_t nv = state_->get_nv();
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(),nv);
    // const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());
    // onst Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(4);

    Data *d = static_cast<Data *>(data.get());
    actuation_->calcDiff(d->multibody.actuation, x, u);

    d->Fx.bottomRightCorner(8, 8).topLeftCorner(4, 4) << 0.0, 1.0, 0.0, 0.0, 11.9411, 0.0, -11.9411, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    d->Fx.bottomRightCorner(4, 4) << 0.0, 1.0, 0.0, 0.0, 11.9411, 0.0, -11.9411, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

    // d->Fx.block(0, state_->get_nv(), state_->get_nv(), state_->get_nv()).setIdentity();
    d->Fu.topLeftCorner(nu_, nu_).setIdentity();
    d->Fu.bottomRightCorner(8, 4).topLeftCorner(4, 2) << 0.0, 0.0, 0.0, -1.0 / 78.8188, 1.0, 0.0, 0.0, 1.0;
    d->Fu.bottomRightCorner(4, 2) << 0.0, 0.0, 0.0, 1.0 / 78.8188, 1.0, 0.0, 0.0, 1.0;

    /*
      std::cout << "d->Fx" << std::endl;
      std::cout << d->Fx << std::endl;

      std::cout << "d->Fu" << std::endl;
      std::cout << d->Fu << std::endl;
    */
    costs_->calcDiff(d->costs, x, u);
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::calcDiff(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, const Eigen::Ref<const VectorXs> &x)
  {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    Data *d = static_cast<Data *>(data.get());
    costs_->calcDiff(d->costs, x);
  }

  template <typename Scalar>
  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar>>
  DifferentialActionModelKinoDynamicsTpl<Scalar>::createData()
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  template <typename Scalar>
  bool DifferentialActionModelKinoDynamicsTpl<Scalar>::checkData(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data)
  {
    boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
    if (d != NULL)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::quasiStatic(
      const boost::shared_ptr<DifferentialActionDataAbstract> &data, Eigen::Ref<VectorXs> u,
      const Eigen::Ref<const VectorXs> &x, const std::size_t, const Scalar)
  {
    if (static_cast<std::size_t>(u.size()) != nu_ + 4)
    {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    if (static_cast<std::size_t>(x.size()) != state_->get_nx() + 8)
    {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    // Static casting the data
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

    const std::size_t nq = state_->get_nq();
    const std::size_t nv = state_->get_nv();

    // Check the velocity input is zero
    assert_pretty(x.segment(nq, nv).isZero(), "The velocity input should be zero for quasi-static to work.");

    d->tmp_xstatic.head(nq) = q;
    d->tmp_xstatic.segment(nq, nv).setZero();
    u.setZero();

    pinocchio::rnea(pinocchio_, d->pinocchio, q, d->tmp_xstatic.segment(nq, nv), d->tmp_xstatic.segment(nq, nv));
    actuation_->calc(d->multibody.actuation, d->tmp_xstatic, u);
    actuation_->calcDiff(d->multibody.actuation, d->tmp_xstatic, u);

    u.noalias() = pseudoInverse(d->multibody.actuation->dtau_du) * d->pinocchio.tau;
    d->pinocchio.tau.setZero();
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::print(std::ostream &os) const
  {
    os << "DifferentialActionModelKinoDynamics {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx()
       << ", nu=" << nu_ << "}";
  }

  template <typename Scalar>
  pinocchio::ModelTpl<Scalar> &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_pinocchio() const
  {
    return pinocchio_;
  }

  template <typename Scalar>
  const boost::shared_ptr<ActuationModelAbstractTpl<Scalar>> &
  DifferentialActionModelKinoDynamicsTpl<Scalar>::get_actuation() const
  {
    return actuation_;
  }

  template <typename Scalar>
  const boost::shared_ptr<CostModelSumTpl<Scalar>> &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_costs()
      const
  {
    return costs_;
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::VectorXs &DifferentialActionModelKinoDynamicsTpl<Scalar>::get_armature() const
  {
    return armature_;
  }

  template <typename Scalar>
  void DifferentialActionModelKinoDynamicsTpl<Scalar>::set_armature(const VectorXs &armature)
  {
    if (static_cast<std::size_t>(armature.size()) != state_->get_nv())
    {
      throw_pretty("Invalid argument: "
                   << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
    }

    armature_ = armature;
    without_armature_ = false;
  }
} // namespace crocoddyl
