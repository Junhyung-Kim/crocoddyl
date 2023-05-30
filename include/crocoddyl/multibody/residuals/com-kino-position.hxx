///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/com-kino-position.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

namespace crocoddyl
{

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state, const Vector3s& cref, const std::size_t nu)
      : Base(state, 3, nu, true, false, true, false, false), cref_(cref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::ResidualModelCoMKinoPositionTpl(boost::shared_ptr<StateKinodynamic> state, const Vector3s& cref)
      : Base(state, 3, true, false, true, false, false), cref_(cref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualModelCoMKinoPositionTpl<Scalar>::~ResidualModelCoMKinoPositionTpl() {}

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                     const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u)
  {
    // Compute the residual residual give the reference CoMPosition position
    Data *d = static_cast<Data *>(data.get());
    const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> x_state = x.tail(8);
    data->r(0) = d->pinocchio->com[0](0) - x_state(0);
    data->r(1) = d->pinocchio->com[0](1) - x_state(4);
    data->r(2) = d->pinocchio->com[0](2) - cref_(2);
    //cref_ = data->r;
  }

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                         const Eigen::Ref<const VectorXs> &x,
                                                         const Eigen::Ref<const VectorXs> &u)
  {
    Data *d = static_cast<Data *>(data.get());
    //const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());

    // Compute the derivatives of the frame placement
    const std::size_t nv = state_->get_nv();
    data->Rx.leftCols(nv) = d->pinocchio->Jcom.block(0, 0, 3, nv);
    (data->Rx.rightCols(8)).leftCols(1)(0) = -1.0;
    (data->Rx.rightCols(4)).leftCols(1)(1) = -1.0;
    //(data->Rx.rightCols(4)).leftCols(1) = -1 * (data->Rx.rightCols(4)).leftCols(1);
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualModelCoMKinoPositionTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualModelCoMKinoPositionTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualModelCoMPosition {cref=" << cref_.transpose().format(fmt) << "}";
  }


template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s& ResidualModelCoMKinoPositionTpl<Scalar>::get_reference() const {
  return cref_;
}

template <typename Scalar>
void ResidualModelCoMKinoPositionTpl<Scalar>::set_reference(const Vector3s& cref) {
  cref_ = cref;
}
} // namespace crocoddyl

