///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/kinoframe-rotation.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualKinoFrameRotationTpl<Scalar>::ResidualKinoFrameRotationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Matrix3s& Rref, const std::size_t nu)
    : Base(state, 3, nu, true, false, false, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualKinoFrameRotationTpl<Scalar>::ResidualKinoFrameRotationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                     const pinocchio::FrameIndex id,
                                                                     const Matrix3s& Rref)
    : Base(state, 3, true, false, false, false, false),
      id_(id),
      Rref_(Rref),
      oRf_inv_(Rref.transpose()),
      pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
ResidualKinoFrameRotationTpl<Scalar>::~ResidualKinoFrameRotationTpl() {}

template <typename Scalar>
void ResidualKinoFrameRotationTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                 const Eigen::Ref<const VectorXs>&,
                                                 const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());
  pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
  d->rRf.noalias() = oRf_inv_ * d->pinocchio->oMf[id_].rotation();
  //data->r = pinocchio::log3(d->rRf);
 // Rref_(0,0) = data->r(0);
 // Rref_(1,1) = data->r(1);
 // Rref_(2,2) = data->r(2);
}

template <typename Scalar>
void ResidualKinoFrameRotationTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data,
                                                     const Eigen::Ref<const VectorXs>&,
                                                     const Eigen::Ref<const VectorXs>&) {
  Data* d = static_cast<Data*>(data.get());

  // Compute the frame Jacobian at the error point
  pinocchio::Jlog3(d->rRf, d->rJf);
  pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::LOCAL, d->fJf);

  // Compute the derivatives of the frame rotation
  const std::size_t nv = state_->get_nv();
  data->Rx.leftCols(nv).noalias() = d->rJf * d->fJf.template bottomRows<3>();
}

template <typename Scalar>
boost::shared_ptr<ResidualDataAbstractTpl<Scalar> > ResidualKinoFrameRotationTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
void ResidualKinoFrameRotationTpl<Scalar>::print(std::ostream& os) const {
  const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
  typename pinocchio::SE3Tpl<Scalar>::Quaternion qref;
  pinocchio::quaternion::assignQuaternion(qref, Rref_);
  os << "ResidualKinoFrameRotation {frame=" << pin_model_->frames[id_].name
     << ", qref=" << qref.coeffs().transpose().format(fmt) << "}";
}

template <typename Scalar>
pinocchio::FrameIndex ResidualKinoFrameRotationTpl<Scalar>::get_id() const {
  return id_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Matrix3s& ResidualKinoFrameRotationTpl<Scalar>::get_reference() const {
  return Rref_;
}

template <typename Scalar>
void ResidualKinoFrameRotationTpl<Scalar>::set_id(const pinocchio::FrameIndex id) {
  id_ = id;
}

template <typename Scalar>
void ResidualKinoFrameRotationTpl<Scalar>::set_reference(const Matrix3s& rotation) {
  Rref_ = rotation;
  oRf_inv_ = rotation.transpose();
}

}  // namespace crocoddyl
