///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/frames.hpp>
#include "crocoddyl/multibody/residuals/kinoframe-translation.hpp"

namespace crocoddyl
{

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s &xref, const std::size_t nu)
      : Base(state, 3, nu, true, false, false, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::ResidualKinoFrameTranslationTpl(boost::shared_ptr<StateKinodynamic> state,
                                                                           const pinocchio::FrameIndex id,
                                                                           const Vector3s &xref)
      : Base(state, 3, true, false, false, false, false), id_(id), xref_(xref), pin_model_(state->get_pinocchio()) {}

  template <typename Scalar>
  ResidualKinoFrameTranslationTpl<Scalar>::~ResidualKinoFrameTranslationTpl() {}

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::calc(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                     const Eigen::Ref<const VectorXs> &x,
                                                     const Eigen::Ref<const VectorXs> &u)
  {
    pinocchio::updateFramePlacement(*pin_model_.get(), *d->pinocchio, id_);
    // Compute the frame translation w.r.t. the reference frame
    Data *d = static_cast<Data *>(data.get());
  //  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    data->r = d->pinocchio->oMf[id_].translation() - xref_;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data,
                                                         const Eigen::Ref<const VectorXs> &x,
                                                         const Eigen::Ref<const VectorXs> &)
  {
    Data *d = static_cast<Data *>(data.get());

    // Compute the derivatives of the frame translation
    const std::size_t nv = state_->get_nv();
    //const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
    pinocchio::getFrameJacobian(*pin_model_.get(), *d->pinocchio, id_, pinocchio::WORLD, d->fJf);
    d->Rx.leftCols(nv).noalias() = d->pinocchio->oMf[id_].rotation() * d->fJf.template topRows<3>();
  }

  template <typename Scalar>
  boost::shared_ptr<ResidualDataAbstractTpl<Scalar>> ResidualKinoFrameTranslationTpl<Scalar>::createData(
      DataCollectorAbstract *const data)
  {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::print(std::ostream &os) const
  {
    const Eigen::IOFormat fmt(2, Eigen::DontAlignCols, ", ", ";\n", "", "", "[", "]");
    os << "ResidualKinoFrameTranslation {frame=" << pin_model_->frames[id_].name
       << ", tref=" << xref_.transpose().format(fmt) << "}";
  }

  template <typename Scalar>
  pinocchio::FrameIndex ResidualKinoFrameTranslationTpl<Scalar>::get_id() const
  {
    return id_;
  }

  template <typename Scalar>
  const typename MathBaseTpl<Scalar>::Vector3s &ResidualKinoFrameTranslationTpl<Scalar>::get_reference() const
  {
    return xref_;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::set_id(const pinocchio::FrameIndex id)
  {
    id_ = id;
  }

  template <typename Scalar>
  void ResidualKinoFrameTranslationTpl<Scalar>::set_reference(const Vector3s &translation)
  {
    xref_ = translation;
  }

} // namespace crocoddyl
