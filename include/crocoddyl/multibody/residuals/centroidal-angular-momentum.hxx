///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "crocoddyl/multibody/residuals/centroidal-angular-momentum.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/math.hpp"
#include "crocoddyl/multibody/actions/kinodyn.hpp"
#include <pinocchio/algorithm/centroidal-derivatives.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <time.h>

namespace crocoddyl
{

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                                               const Vector6s &href,
                                                                                               const std::size_t nu)
      : Base(state, 2, nu, true, true, true, false, false), href_(href), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::ResidualModelCentroidalAngularMomentumTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                                               const std::size_t nu)
      : Base(state, 2, nu, true, true, true, false, false), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCentroidalAngularMomentumTpl<Scalar>::~ResidualModelCentroidalAngularMomentumTpl() {}

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                               const Eigen::Ref<const VectorXs> &x,
                                                               const Eigen::Ref<const VectorXs> &u)
  {
    time_t start,start1, end;
    // Compute the residual residual give the reference centroidal momentum
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    
    pinocchio::computeCentroidalMomentum(*pin_model_.get(), *d->pinocchio, q, v);  
    data->r(0) = d->pinocchio->hg.toVector()(3) - x_state(7);
    data->r(1) = d->pinocchio->hg.toVector()(4) - x_state(3);
    href_(0) = data->r(0);
    href_(1) = data->r(1);
  }

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                                   const Eigen::Ref<const VectorXs> &x,
                                                                   const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());
    const std::size_t &nv = state_->get_nv();
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.segment(state_->get_nq(), state_->get_nv());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> a = u.head(state_->get_nv());
    //pinocchio::computeRNEADerivatives(*pin_model_.get(), *d->pinocchio, q, v, a);
    pinocchio::computeCentroidalDynamicsDerivatives(*pin_model_.get(), *d->pinocchio,  q, v, a, d->dh_dq, d->dhd_dq, d->dhd_dv, d->dhd_da);
    data->Rx.rightCols(1)(0) = -1;
    data->Rx.rightCols(5).leftCols(1)(1) = -1;
    data->Rx.leftCols(nv) = d->dh_dq.block(3, 0, 2, nv);
    data->Rx.block(0, nv, 2, nv) = d->dhd_da.block(3, 0, 2, nv);
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualModelCentroidalAngularMomentumTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualModelCentroidalAngularMomentum {href=" << href_.transpose().format(fmt) << "}";
  }
  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::Vector6s& ResidualModelCentroidalAngularMomentumTpl<Scalar>::get_reference() const {
    return href_;
  }

  template <typename Scalar>
  void ResidualModelCentroidalAngularMomentumTpl<Scalar>::set_reference(const Vector6s& href) {
    href_ = href;
  }
} // namespace crocoddyl